## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 7200 seconds
Split limit: 100
Threshold: 51.0142900446


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898)
1: (-36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940)
2: (-32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537)
3: (-35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419)
4: (-41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263)
5: (-37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252)
6: (-62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132)
7: (-44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597)
8: (-50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352)
9: (-40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131)
10: (-63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549)
11: (-59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469)
12: (-60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122)
13: (-65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675)
14: (-99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209)
15: (-47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354)
16: (-62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391)
17: (-96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654)
18: (-59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035)
19: (-48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014)
20: (-46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262)
21: (-58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035)
22: (-61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466)
23: (-47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099)
24: (-57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676)
25: (-51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661)
26: (-70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164)
27: (-57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348)
28: (-47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574)
29: (-60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737)
30: (-58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578)
31: (-59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628)
32: (-61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560)
33: (-86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329)
34: (-75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727)
35: (-71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940)
36: (-71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631)
37: (-102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932)
38: (-87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134)
39: (-97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755)
40: (-78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759)
41: (-64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865)
42: (-48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.87 + 110.43 = 113.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -51.0653554, upper bound: 51.0653554

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0645988, upper bound: 51.0301099
time: 93.06 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0645988, upper bound: 51.0645987
time: 118.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 211.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 211.86
Output dim: 1, lower bound: -51.0645988, upper bound: 51.0301099
IS_A2, status: Status.UNKNOWN, split count: 1, time: 211.86
Output dim: 1, lower bound: -51.0645988, upper bound: 51.0645987

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -68.8715744, 38.8136749, -68.9073181, 38.8510590, -107.7226334, 107.7209854
1: -36.3955421, 36.5484619, -36.4117279, 36.5893555, -72.9848938, 72.9601898
2: -32.1530685, 38.5954742, -32.1661873, 38.6427383, -70.7958069, 70.7616577
3: -35.6792488, 42.9942856, -35.6909904, 43.0332336, -78.7124786, 78.6852722
4: -41.6690559, 40.2607117, -41.6866379, 40.2974548, -81.9664993, 81.9473495
5: -37.1778984, 42.4139824, -37.1906509, 42.4665527, -79.6444550, 79.6046295
6: -62.4035721, 37.6039963, -62.4253540, 37.6265106, -100.0300827, 100.0293427
7: -44.4081078, 40.4295502, -44.4234047, 40.4915009, -84.8996124, 84.8529510
8: -50.0212784, 46.5883865, -50.0409241, 46.6334724, -96.6547394, 96.6293106
9: -40.9354095, 43.8608627, -40.9549217, 43.9149132, -84.8503265, 84.8157806
10: -63.2606239, 58.4469719, -63.2825089, 58.4951286, -121.7557373, 121.7294769
11: -59.3824310, 33.7012100, -59.4149361, 33.7171135, -93.0995331, 93.1161499
12: -60.7804031, 42.4675179, -60.7989616, 42.4934387, -103.2738342, 103.2664795
13: -65.4895630, 61.3020859, -65.5121536, 61.3596725, -126.8492355, 126.8142395
14: -99.5043564, 46.6643257, -99.5405731, 46.7101936, -146.2145386, 146.2048950
15: -47.8686104, 43.1285362, -47.8929977, 43.1452179, -91.0138245, 91.0215302
16: -62.6425056, 45.7681885, -62.6734924, 45.8313065, -108.4738007, 108.4416656
17: -96.4193039, 43.8177795, -96.4504776, 43.8523331, -140.2716370, 140.2682495
18: -59.4012222, 47.9822540, -59.4452438, 47.9971542, -107.3983765, 107.4274979
19: -48.5492821, 28.0006523, -48.6100960, 28.0086708, -76.5579529, 76.6107483
20: -46.5068130, 32.1664581, -46.5590706, 32.1801414, -78.6869507, 78.7255249
21: -58.2582359, 32.8014297, -58.3146896, 32.8124428, -91.0706711, 91.1161194
22: -60.9192619, 34.6801834, -61.0115662, 34.6914902, -95.6107483, 95.6917419
23: -47.1166267, 35.6743011, -47.1693268, 35.6863937, -82.8030243, 82.8436279
24: -57.3415985, 34.0609207, -57.4100914, 34.0700531, -91.4116516, 91.4710083
25: -51.4816666, 37.4178772, -51.5581627, 37.4311218, -88.9127884, 88.9760437
26: -69.9837494, 50.3404083, -70.0516357, 50.3543243, -120.3380737, 120.3920441
27: -56.9107552, 38.9458008, -56.9821930, 38.9570427, -95.8677979, 95.9279938
28: -47.5649033, 39.3469810, -47.6367455, 39.3596802, -86.9245834, 86.9837265
29: -60.1168671, 30.6189537, -60.1981430, 30.6297340, -90.7465744, 90.8170929
30: -58.4313164, 40.5697632, -58.4698906, 40.5878830, -99.0191956, 99.0396500
31: -59.4253159, 34.4449310, -59.4986534, 34.4572525, -93.8825607, 93.9435730
32: -61.0571442, 35.8793259, -61.0883369, 35.8959846, -96.9531250, 96.9676666
33: -86.4392548, 46.4342728, -86.4876709, 46.4538078, -132.8930664, 132.9219360
34: -75.0847473, 32.0237656, -75.1511002, 32.0372849, -107.1220245, 107.1748657
35: -70.8660431, 35.2762184, -70.9346313, 35.2887497, -106.1547775, 106.2108459
36: -71.7721252, 37.9489861, -71.8491898, 37.9595795, -109.7317047, 109.7981720
37: -102.1545944, 33.5243835, -102.2124329, 33.5364456, -135.6910400, 135.7368164
38: -86.8673553, 50.8318558, -86.9674835, 50.8525620, -137.7199097, 137.7993317
39: -97.7883759, 44.0501442, -97.8480682, 44.0651283, -141.8535004, 141.8982086
40: -78.4653931, 34.5183296, -78.5032501, 34.5366325, -113.0020294, 113.0215759
41: -64.6108475, 40.8830338, -64.6461639, 40.8967667, -105.5076141, 105.5291824
42: -48.5285034, 36.2195625, -48.5499191, 36.2381783, -84.7666779, 84.7694855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=436, inp2_unstable=437, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0114332
time: 183.87 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0267565
time: 96.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -69.1882553, 38.9037132, -68.9302521, 38.8823433, -108.0706024, 107.8339615
1: -36.6125908, 36.6365891, -36.4219856, 36.6254044, -73.2379913, 73.0585785
2: -32.3273926, 38.6992264, -32.1732407, 38.6845169, -71.0119095, 70.8724594
3: -35.8297195, 43.0879135, -35.6973419, 43.0671349, -78.8968506, 78.7852554
4: -41.8762169, 40.3450775, -41.6960106, 40.3292923, -82.2055054, 82.0410919
5: -37.3228226, 42.5344620, -37.1965675, 42.5124779, -79.8352966, 79.7310257
6: -62.4720421, 37.7276382, -62.4398689, 37.6394043, -100.1114349, 100.1674957
7: -44.6341553, 40.5591965, -44.4317398, 40.5461578, -85.1803131, 84.9909363
8: -50.2836380, 46.6986809, -50.0534668, 46.6722908, -96.9559326, 96.7521362
9: -41.1283455, 43.9803581, -40.9684258, 43.9622879, -85.0906372, 84.9487839
10: -63.4740410, 58.5672417, -63.2972527, 58.5344315, -122.0084686, 121.8644943
11: -59.5228539, 33.7889175, -59.4401093, 33.7258949, -93.2487488, 93.2290268
12: -60.8616562, 42.6074257, -60.8119431, 42.5120392, -103.3736954, 103.4193726
13: -65.6543121, 61.4548454, -65.5230103, 61.4093475, -127.0636520, 126.9778519
14: -99.7215805, 46.7730980, -99.5615692, 46.7487946, -146.4703674, 146.3346558
15: -48.0145798, 43.2174110, -47.9094963, 43.1574631, -91.1720428, 91.1268997
16: -62.8846283, 45.9081154, -62.6958275, 45.8871765, -108.7718048, 108.6039276
17: -96.6565857, 43.9177170, -96.4677734, 43.8806992, -140.5372772, 140.3854980
18: -59.5311737, 48.1333694, -59.4819031, 48.0064430, -107.5376129, 107.6152725
19: -48.6990204, 28.1367226, -48.6631241, 28.0118484, -76.7108688, 76.7998428
20: -46.6273575, 32.3022957, -46.6040688, 32.1868820, -78.8142395, 78.9063644
21: -58.4167290, 32.9303970, -58.3626442, 32.8175392, -91.2342682, 91.2930298
22: -61.1357460, 34.8428726, -61.0922852, 34.6986008, -95.8343506, 95.9351578
23: -47.2439117, 35.7970009, -47.2148476, 35.6925430, -82.9364548, 83.0118408
24: -57.5013962, 34.1761932, -57.4691849, 34.0741501, -91.5755463, 91.6453781
25: -51.6544304, 37.5904770, -51.6254654, 37.4389038, -89.0933380, 89.2159424
26: -70.1526413, 50.5184326, -70.1105499, 50.3604126, -120.5130463, 120.6289825
27: -57.0862732, 39.0863266, -57.0440140, 38.9617004, -96.0479660, 96.1303406
28: -47.7209091, 39.5228806, -47.6999817, 39.3654823, -87.0863876, 87.2228622
29: -60.3229828, 30.7603989, -60.2685890, 30.6360321, -90.9590073, 91.0289841
30: -58.5376854, 40.6729279, -58.4995155, 40.5989227, -99.1366119, 99.1724396
31: -59.6095543, 34.6339493, -59.5629807, 34.4639397, -94.0734940, 94.1969299
32: -61.1467819, 35.9917755, -61.1109467, 35.9062271, -97.0529938, 97.1027069
33: -86.5788651, 46.5739441, -86.5260925, 46.4670181, -133.0458679, 133.1000366
34: -75.2390747, 32.2006912, -75.2062912, 32.0453110, -107.2843781, 107.4069824
35: -71.0292664, 35.4508820, -70.9944458, 35.2962112, -106.3254776, 106.4453049
36: -71.9454422, 38.1606140, -71.9154434, 37.9657097, -109.9111176, 110.0760574
37: -102.3193283, 33.6013870, -102.2587509, 33.5437469, -135.8630524, 135.8601379
38: -87.0936432, 51.1167870, -87.0521393, 50.8659782, -137.9596252, 138.1689301
39: -97.9601974, 44.1535645, -97.8947067, 44.0744476, -142.0346375, 142.0482788
40: -78.6092377, 34.5949478, -78.5328522, 34.5442886, -113.1535263, 113.1277924
41: -64.7176361, 40.9701462, -64.6744614, 40.9025879, -105.6202240, 105.6446075
42: -48.5966301, 36.2983475, -48.5666656, 36.2463799, -84.8430023, 84.8650131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=436, inp2_unstable=437, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0278149
time: 110.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0635915
time: 98.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 211.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 211.65
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0114332
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 211.65
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0267565
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 211.65
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0278149
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 211.65
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0635915

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -68.6198730, 38.7837296, -68.3780441, 38.7514496, -107.3713074, 107.1617584
1: -36.2367249, 36.5346527, -36.0793152, 36.5188141, -72.7555389, 72.6139603
2: -32.0181198, 38.5803757, -31.8855286, 38.5666771, -70.5847931, 70.4658966
3: -35.5559921, 42.9709663, -35.4333878, 42.9450188, -78.5010071, 78.4043503
4: -41.4906540, 40.2419662, -41.3125305, 40.2252693, -81.7159271, 81.5544891
5: -37.0430260, 42.3897362, -36.9088135, 42.3565636, -79.3995895, 79.2985535
6: -62.3629150, 37.4837379, -62.3248138, 37.3732147, -99.7361298, 99.8085480
7: -44.2328453, 40.4123535, -44.0606728, 40.3908005, -84.6236420, 84.4730225
8: -49.8378944, 46.5624390, -49.6551132, 46.5530434, -96.3909378, 96.2175522
9: -40.7762985, 43.8392830, -40.6188469, 43.8093948, -84.5856781, 84.4581299
10: -63.0997810, 58.4061317, -62.9422073, 58.3628731, -121.4626389, 121.3483276
11: -59.3113098, 33.6181145, -59.2533417, 33.5378265, -92.8491364, 92.8714600
12: -60.7321243, 42.3506470, -60.6867065, 42.2410011, -102.9731293, 103.0373535
13: -65.3159027, 61.2511368, -65.1494064, 61.1883011, -126.5042038, 126.4005280
14: -99.2973404, 46.6372643, -99.1007843, 46.6014442, -145.8987885, 145.7380524
15: -47.7668648, 43.0480881, -47.6701088, 42.9759750, -90.7428207, 90.7182007
16: -62.4702911, 45.7418289, -62.3089790, 45.6927567, -108.1630402, 108.0508041
17: -96.2264938, 43.7805634, -96.0425186, 43.7495041, -139.9759979, 139.8230896
18: -59.3539391, 47.8744431, -59.3177719, 47.7690735, -107.1229935, 107.1922150
19: -48.5149574, 27.8824997, -48.4877281, 27.7660847, -76.2810440, 76.3702240
20: -46.4762268, 32.0566254, -46.4554787, 31.9529800, -78.4291992, 78.5121002
21: -58.2047310, 32.6903992, -58.1671791, 32.5821495, -90.7868805, 90.8575745
22: -60.8753777, 34.5600357, -60.8446083, 34.4437675, -95.3191376, 95.4046478
23: -47.0855751, 35.5487938, -47.0483665, 35.4242401, -82.5098114, 82.5971603
24: -57.2971878, 33.9583473, -57.2677994, 33.8560982, -91.1532745, 91.2261505
25: -51.4479561, 37.2970581, -51.4353180, 37.1795425, -88.6274872, 88.7323685
26: -69.9320984, 50.1724510, -69.8635635, 50.0088501, -119.9409485, 120.0360107
27: -56.8640938, 38.8174133, -56.8247604, 38.6899872, -95.5540771, 95.6421738
28: -47.5382729, 39.1964760, -47.5089760, 39.0475388, -86.5858154, 86.7054520
29: -60.0663071, 30.5030003, -60.0291138, 30.3883209, -90.4546280, 90.5321121
30: -58.3772240, 40.4972610, -58.3427658, 40.4308701, -98.8080902, 98.8400192
31: -59.3799591, 34.3030968, -59.3552589, 34.1624069, -93.5423660, 93.6583405
32: -61.0138016, 35.7805786, -60.9803772, 35.6883926, -96.7021790, 96.7609558
33: -86.3782730, 46.3150215, -86.3126526, 46.1948090, -132.5730896, 132.6276703
34: -75.0487366, 31.8777122, -75.0021515, 31.7360954, -106.7848358, 106.8798599
35: -70.8231506, 35.1471596, -70.7786331, 35.0174713, -105.8406219, 105.9257889
36: -71.7389832, 37.7963638, -71.7019730, 37.6419029, -109.3808746, 109.4983368
37: -102.0927200, 33.4156151, -102.0122757, 33.3098373, -135.4025574, 135.4278870
38: -86.8245087, 50.6345940, -86.7696075, 50.4419785, -137.2664795, 137.4042053
39: -97.7144165, 43.9866142, -97.6548462, 43.9233742, -141.6377869, 141.6414490
40: -78.4039993, 34.4367561, -78.3517914, 34.3646660, -112.7686539, 112.7885437
41: -64.5669250, 40.7545853, -64.5093689, 40.6301270, -105.1970444, 105.2639542
42: -48.4975624, 36.1059265, -48.4663239, 36.0003548, -84.4979095, 84.5722504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=436, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0194119, upper bound: 51.0047181
time: 105.19 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0106063
time: 91.58 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -68.8657532, 38.8125801, -68.8916550, 38.8482132, -107.7139664, 107.7042160
1: -36.3911133, 36.5478516, -36.4005966, 36.5877914, -72.9789047, 72.9484406
2: -32.1491699, 38.5947723, -32.1560020, 38.6408844, -70.7900543, 70.7507782
3: -35.6757202, 42.9932671, -35.6816330, 43.0306435, -78.7063522, 78.6748962
4: -41.6641388, 40.2598114, -41.6732597, 40.2951202, -81.9592514, 81.9330750
5: -37.1746902, 42.4130287, -37.1819992, 42.4640045, -79.6386948, 79.5950241
6: -62.4015503, 37.5992012, -62.4199066, 37.6135101, -100.0150299, 100.0191040
7: -44.4034767, 40.4288254, -44.4123688, 40.4896698, -84.8931427, 84.8411942
8: -50.0162430, 46.5872040, -50.0273552, 46.6303749, -96.6466141, 96.6145630
9: -40.9322090, 43.8597603, -40.9462662, 43.9119835, -84.8441925, 84.8060303
10: -63.2571640, 58.4452477, -63.2732964, 58.4906082, -121.7477722, 121.7185287
11: -59.3786469, 33.6975021, -59.4051361, 33.7073708, -93.0860062, 93.1026382
12: -60.7782440, 42.4634933, -60.7935257, 42.4826317, -103.2608795, 103.2570190
13: -65.4853745, 61.3000526, -65.5009155, 61.3543587, -126.8397064, 126.8009491
14: -99.4992371, 46.6633377, -99.5266953, 46.7076187, -146.2068481, 146.1900330
15: -47.8647079, 43.1260986, -47.8826981, 43.1395721, -91.0042801, 91.0087967
16: -62.6382599, 45.7670860, -62.6622581, 45.8285446, -108.4667969, 108.4293442
17: -96.4140854, 43.8165321, -96.4367218, 43.8491135, -140.2631989, 140.2532349
18: -59.3989792, 47.9795532, -59.4392624, 47.9898720, -107.3888397, 107.4188080
19: -48.5479279, 27.9972839, -48.6065445, 27.9995632, -76.5474854, 76.6038284
20: -46.5054855, 32.1627617, -46.5556717, 32.1712189, -78.6766968, 78.7184296
21: -58.2560081, 32.7978859, -58.3089409, 32.8029251, -91.0589218, 91.1068268
22: -60.9177208, 34.6766510, -61.0076180, 34.6819305, -95.5996399, 95.6842651
23: -47.1150322, 35.6706772, -47.1651611, 35.6765289, -82.7915649, 82.8358383
24: -57.3392029, 34.0578918, -57.4038734, 34.0618439, -91.4010391, 91.4617615
25: -51.4800110, 37.4140701, -51.5539131, 37.4208565, -88.9008636, 88.9679871
26: -69.9815445, 50.3366928, -70.0459137, 50.3443909, -120.3259354, 120.3826065
27: -56.9088821, 38.9420090, -56.9771614, 38.9468002, -95.8556824, 95.9191742
28: -47.5635452, 39.3426437, -47.6331673, 39.3479462, -86.9114838, 86.9758072
29: -60.1151237, 30.6156864, -60.1937256, 30.6208496, -90.7359695, 90.8094101
30: -58.4279175, 40.5672836, -58.4608841, 40.5812149, -99.0091324, 99.0281677
31: -59.4236336, 34.4409409, -59.4943428, 34.4466133, -93.8702393, 93.9352798
32: -61.0555229, 35.8741646, -61.0840187, 35.8830528, -96.9385681, 96.9581680
33: -86.4370422, 46.4312897, -86.4817352, 46.4460297, -132.8830719, 132.9130249
34: -75.0829163, 32.0208893, -75.1462250, 32.0297127, -107.1126251, 107.1670990
35: -70.8640213, 35.2734680, -70.9292603, 35.2814903, -106.1455078, 106.2027206
36: -71.7708130, 37.9460983, -71.8457184, 37.9519386, -109.7227478, 109.7918015
37: -102.1519699, 33.5214462, -102.2055740, 33.5287323, -135.6806946, 135.7270203
38: -86.8655701, 50.8276901, -86.9626541, 50.8414803, -137.7070465, 137.7903442
39: -97.7856140, 44.0468025, -97.8406219, 44.0561752, -141.8417969, 141.8874207
40: -78.4630890, 34.5130577, -78.4971008, 34.5226135, -112.9857025, 113.0101547
41: -64.6091461, 40.8786240, -64.6416626, 40.8849144, -105.4940491, 105.5202866
42: -48.5273056, 36.2156563, -48.5468254, 36.2275085, -84.7548141, 84.7624817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=436, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0194699, upper bound: 51.0202940
time: 104.17 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0259217
time: 110.34 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -68.9383469, 38.8741455, -68.4022064, 38.7839584, -107.7223053, 107.2763443
1: -36.4530563, 36.6230621, -36.0898819, 36.5553474, -73.0084076, 72.7129440
2: -32.1921082, 38.6843262, -31.8931503, 38.6089134, -70.8010254, 70.5774765
3: -35.7063141, 43.0647697, -35.4403534, 42.9790993, -78.6854095, 78.5051270
4: -41.6980095, 40.3263779, -41.3233643, 40.2568016, -81.9548111, 81.6497421
5: -37.1877441, 42.5105591, -36.9156837, 42.4027863, -79.5905304, 79.4262390
6: -62.4311371, 37.6103668, -62.3397675, 37.3875122, -99.8186493, 99.9501343
7: -44.4579391, 40.5421677, -44.0693550, 40.4453621, -84.9033051, 84.6115189
8: -50.0999451, 46.6729050, -49.6685524, 46.5921288, -96.6920624, 96.3414536
9: -40.9707451, 43.9588013, -40.6334267, 43.8583565, -84.8291016, 84.5922241
10: -63.3135109, 58.5269432, -62.9582062, 58.4035721, -121.7170792, 121.4851532
11: -59.4507828, 33.7050896, -59.2781219, 33.5482101, -92.9989929, 92.9832153
12: -60.8124504, 42.4937744, -60.7006302, 42.2606926, -103.0731354, 103.1944046
13: -65.4819565, 61.4045982, -65.1626434, 61.2359276, -126.7178650, 126.5672455
14: -99.5159760, 46.7464256, -99.1237335, 46.6397820, -146.1557617, 145.8701477
15: -47.9134407, 43.1367836, -47.6882286, 42.9883614, -90.9017944, 90.8250122
16: -62.7130165, 45.8818817, -62.3313751, 45.7499542, -108.4629517, 108.2132568
17: -96.4649658, 43.8807411, -96.0614624, 43.7774506, -140.2424011, 139.9421997
18: -59.4844437, 48.0258331, -59.3546638, 47.7793961, -107.2638321, 107.3804855
19: -48.6649933, 28.0186462, -48.5409164, 27.7703018, -76.4352951, 76.5595551
20: -46.5972748, 32.1928253, -46.5010300, 31.9608421, -78.5581207, 78.6938553
21: -58.3636818, 32.8195953, -58.2153931, 32.5885963, -90.9522781, 91.0349884
22: -61.0921440, 34.7223473, -60.9257851, 34.4513206, -95.5434570, 95.6481323
23: -47.2131195, 35.6719131, -47.0937843, 35.4317245, -82.6448441, 82.7657013
24: -57.4573517, 34.0741615, -57.3271751, 33.8614426, -91.3187943, 91.4013367
25: -51.6211967, 37.4694138, -51.5008278, 37.1880760, -88.8092728, 88.9702377
26: -70.1018448, 50.3517685, -69.9241791, 50.0160255, -120.1178741, 120.2759476
27: -57.0402069, 38.9590759, -56.8869019, 38.6964874, -95.7366791, 95.8459778
28: -47.6947708, 39.3728714, -47.5725250, 39.0547218, -86.7494888, 86.9453964
29: -60.2723389, 30.6445713, -60.0995827, 30.3952904, -90.6676331, 90.7441559
30: -58.4844742, 40.6007347, -58.3724747, 40.4434738, -98.9279480, 98.9731979
31: -59.5645027, 34.4920158, -59.4199295, 34.1701431, -93.7346497, 93.9119415
32: -61.1036758, 35.8926315, -61.0033188, 35.6995659, -96.8032379, 96.8959503
33: -86.5186234, 46.4554176, -86.3508301, 46.2083969, -132.7270203, 132.8062439
34: -75.2035522, 32.0567703, -75.0608292, 31.7445412, -106.9480896, 107.1175842
35: -70.9870453, 35.3219604, -70.8390808, 35.0255356, -106.0125732, 106.1610260
36: -71.9127655, 38.0085983, -71.7718735, 37.6486206, -109.5613861, 109.7804718
37: -102.2579346, 33.4907265, -102.0601196, 33.3176041, -135.5755310, 135.5508423
38: -87.0512695, 50.9228745, -86.8606491, 50.4560394, -137.5073090, 137.7835083
39: -97.8871231, 44.0892677, -97.7018433, 43.9329071, -141.8200226, 141.7911072
40: -78.5471725, 34.5141830, -78.3822479, 34.3728485, -112.9200058, 112.8964233
41: -64.6742859, 40.8437119, -64.5385284, 40.6374512, -105.3117371, 105.3822403
42: -48.5652199, 36.1845779, -48.4833603, 36.0104141, -84.5756226, 84.6679230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=436, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0194119, upper bound: 51.0207393
time: 90.05 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0264228
time: 96.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -69.1814117, 38.9025803, -68.9142227, 38.8793869, -108.0607910, 107.8168030
1: -36.6083221, 36.6359482, -36.4107437, 36.6237564, -73.2320786, 73.0466919
2: -32.3236122, 38.6985092, -32.1628456, 38.6826401, -71.0062408, 70.8613586
3: -35.8263359, 43.0869408, -35.6878166, 43.0644951, -78.8908310, 78.7747574
4: -41.8712082, 40.3441925, -41.6823883, 40.3268776, -82.1980896, 82.0265808
5: -37.3189850, 42.5334816, -37.1876907, 42.5098724, -79.8288574, 79.7211685
6: -62.4700813, 37.7222786, -62.4342957, 37.6261177, -100.0961990, 100.1565704
7: -44.6295509, 40.5584564, -44.4205704, 40.5443039, -85.1738586, 84.9790268
8: -50.2787437, 46.6974449, -50.0397110, 46.6691017, -96.9478302, 96.7371521
9: -41.1243820, 43.9792709, -40.9596100, 43.9592934, -85.0836639, 84.9388809
10: -63.4697723, 58.5654678, -63.2878380, 58.5297165, -121.9994812, 121.8532944
11: -59.5191231, 33.7855797, -59.4301872, 33.7157631, -93.2348785, 93.2157593
12: -60.8598251, 42.6026688, -60.8063087, 42.5009804, -103.3608017, 103.4089737
13: -65.6490250, 61.4527054, -65.5113449, 61.4039040, -127.0529327, 126.9640503
14: -99.7153549, 46.7720413, -99.5472488, 46.7461090, -146.4614563, 146.3192902
15: -48.0103760, 43.2146225, -47.8989983, 43.1516380, -91.1620178, 91.1136169
16: -62.8794289, 45.9069939, -62.6844025, 45.8843842, -108.7638092, 108.5914001
17: -96.6504822, 43.9163971, -96.4536667, 43.8773880, -140.5278625, 140.3700562
18: -59.5288315, 48.1302719, -59.4756851, 47.9990005, -107.5278320, 107.6059570
19: -48.6975327, 28.1334343, -48.6594582, 28.0025387, -76.7000656, 76.7928925
20: -46.6259499, 32.2985535, -46.6004868, 32.1776962, -78.8036499, 78.8990402
21: -58.4143105, 32.9269066, -58.3566513, 32.8077927, -91.2221069, 91.2835541
22: -61.1340752, 34.8395538, -61.0882568, 34.6888809, -95.8229523, 95.9278030
23: -47.2422409, 35.7933426, -47.2105484, 35.6824722, -82.9247131, 83.0038910
24: -57.4989624, 34.1731911, -57.4627838, 34.0657501, -91.5647125, 91.6359711
25: -51.6526756, 37.5868034, -51.6210403, 37.4284630, -89.0811386, 89.2078400
26: -70.1502380, 50.5135536, -70.1046143, 50.3501587, -120.5003967, 120.6181641
27: -57.0842323, 39.0825157, -57.0387917, 38.9511909, -96.0354233, 96.1213074
28: -47.7194061, 39.5185585, -47.6961823, 39.3534431, -87.0728455, 87.2147369
29: -60.3211975, 30.7572975, -60.2641068, 30.6270180, -90.9482117, 91.0214081
30: -58.5341148, 40.6703033, -58.4901237, 40.5920029, -99.1260986, 99.1604233
31: -59.6077805, 34.6300812, -59.5585518, 34.4531021, -94.0608826, 94.1886292
32: -61.1452599, 35.9866676, -61.1065216, 35.8929520, -97.0382080, 97.0931854
33: -86.5765381, 46.5708504, -86.5200043, 46.4591255, -133.0356445, 133.0908508
34: -75.2372131, 32.1969223, -75.2012634, 32.0377083, -107.2749176, 107.3981781
35: -71.0271454, 35.4475784, -70.9888763, 35.2888756, -106.3160095, 106.4364548
36: -71.9440155, 38.1567802, -71.9118576, 37.9579582, -109.9019775, 110.0686340
37: -102.3166885, 33.5986557, -102.2516785, 33.5359535, -135.8526459, 135.8503418
38: -87.0917740, 51.1113586, -87.0472794, 50.8546829, -137.9464569, 138.1586304
39: -97.9572830, 44.1501999, -97.8871078, 44.0653725, -142.0226440, 142.0373077
40: -78.6070023, 34.5894508, -78.5265884, 34.5294762, -113.1364746, 113.1160355
41: -64.7160110, 40.9655991, -64.6698837, 40.8903885, -105.6063995, 105.6354752
42: -48.5954819, 36.2935410, -48.5634575, 36.2351723, -84.8306580, 84.8569946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=436, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0194699, upper bound: 51.0567540
time: 84.25 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0622010
time: 171.12 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 257.77 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 257.77
Output dim: 1, lower bound: -51.0194119, upper bound: 51.0047181
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 257.77
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0106063
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 257.77
Output dim: 1, lower bound: -51.0194699, upper bound: 51.0202940
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 257.77
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0259217
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 257.77
Output dim: 1, lower bound: -51.0194119, upper bound: 51.0207393
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 257.77
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0264228
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 257.77
Output dim: 1, lower bound: -51.0194699, upper bound: 51.0567540
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 257.77
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0622010

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -68.1452942, 38.7080688, -68.1568604, 38.7285385, -106.8738251, 106.8649139
1: -35.9278603, 36.4771423, -35.9335899, 36.5086327, -72.4364929, 72.4107361
2: -31.7583046, 38.5147476, -31.7624016, 38.5545959, -70.3128891, 70.2771454
3: -35.3128815, 42.8980598, -35.3181114, 42.9250488, -78.2379227, 78.2161713
4: -41.1662140, 40.1847382, -41.1596413, 40.2094460, -81.3756561, 81.3443756
5: -36.7780151, 42.3067551, -36.7832870, 42.3376732, -79.1156769, 79.0900421
6: -62.2808037, 37.2316322, -62.2962189, 37.2560005, -99.5367889, 99.5278473
7: -43.8806305, 40.3219757, -43.8928680, 40.3752060, -84.2558365, 84.2148438
8: -49.4565239, 46.4839325, -49.4748268, 46.5312805, -95.9878082, 95.9587479
9: -40.4723930, 43.7497330, -40.4761734, 43.7904510, -84.2628479, 84.2259064
10: -62.7727890, 58.2939072, -62.7893944, 58.3283615, -121.1011505, 121.0832977
11: -59.1658707, 33.4921989, -59.1867981, 33.4802856, -92.6461563, 92.6790009
12: -60.6454353, 42.0558777, -60.6535606, 42.1064987, -102.7519226, 102.7094421
13: -65.0424423, 61.1447868, -65.0223694, 61.1467094, -126.1891479, 126.1671448
14: -98.9256439, 46.5455322, -98.9290237, 46.5786896, -145.5043335, 145.4745483
15: -47.5618668, 42.9075890, -47.5778923, 42.9114494, -90.4733124, 90.4854813
16: -62.1415634, 45.6286316, -62.1556396, 45.6690750, -107.8106384, 107.7842712
17: -95.8989639, 43.6878014, -95.8902588, 43.7154236, -139.6143799, 139.5780640
18: -59.2390442, 47.6380539, -59.2775345, 47.6576843, -106.8967285, 106.9155807
19: -48.4234657, 27.6853828, -48.4593582, 27.6714706, -76.0949326, 76.1447372
20: -46.3948517, 31.8609657, -46.4314308, 31.8608551, -78.2557068, 78.2923889
21: -58.0905342, 32.5031738, -58.1228905, 32.4937134, -90.5842285, 90.6260681
22: -60.7401237, 34.3345985, -60.8068581, 34.3363647, -95.0764847, 95.1414566
23: -47.0074425, 35.3563385, -47.0230255, 35.3336105, -82.3410492, 82.3793640
24: -57.2025261, 33.7945328, -57.2348175, 33.7788811, -90.9814072, 91.0293503
25: -51.3413048, 37.0770874, -51.4066315, 37.0755310, -88.4168396, 88.4837189
26: -69.7868042, 49.8569183, -69.8256836, 49.8587608, -119.6455688, 119.6825943
27: -56.7594833, 38.6119080, -56.7872543, 38.5936508, -95.3531342, 95.3991623
28: -47.4501038, 38.9419708, -47.4880447, 38.9276276, -86.3777313, 86.4300156
29: -59.9369583, 30.3097343, -59.9855614, 30.2967758, -90.2337341, 90.2952957
30: -58.2536316, 40.3799057, -58.2891273, 40.3788910, -98.6325226, 98.6690292
31: -59.2644997, 34.0376663, -59.3162651, 34.0362015, -93.3006897, 93.3539276
32: -60.9213486, 35.5600090, -60.9474487, 35.5863647, -96.5077057, 96.5074615
33: -86.2285614, 46.0344238, -86.2660980, 46.0634270, -132.2919922, 132.3005066
34: -74.9198456, 31.5750465, -74.9730148, 31.5923862, -106.5122223, 106.5480652
35: -70.6846008, 34.8513412, -70.7453308, 34.8768539, -105.5614548, 105.5966644
36: -71.6050873, 37.4507599, -71.6753082, 37.4773026, -109.0823898, 109.1260681
37: -101.9216156, 33.1661758, -101.9634399, 33.1914635, -135.1130676, 135.1296082
38: -86.6378632, 50.1831741, -86.7322998, 50.2281075, -136.8659668, 136.9154663
39: -97.5529633, 43.7866058, -97.6014557, 43.8289719, -141.3819275, 141.3880615
40: -78.2810822, 34.2263832, -78.3039551, 34.2659454, -112.5470276, 112.5303268
41: -64.4601440, 40.5263786, -64.4774933, 40.5233231, -104.9834595, 105.0038681
42: -48.4247360, 35.9012222, -48.4427948, 35.9049835, -84.3297119, 84.3440170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1653

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9646235
time: 96.71 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0025596
time: 97.79 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -68.6054840, 38.7819138, -68.3731308, 38.7507935, -107.3562775, 107.1550446
1: -36.2271500, 36.5337029, -36.0760498, 36.5184822, -72.7456360, 72.6097565
2: -32.0099258, 38.5792236, -31.8827362, 38.5662689, -70.5761948, 70.4619598
3: -35.5482941, 42.9693108, -35.4307556, 42.9444618, -78.4927521, 78.4000626
4: -41.4801064, 40.2403450, -41.3089600, 40.2247009, -81.7048035, 81.5493011
5: -37.0346222, 42.3880196, -36.9059525, 42.3559456, -79.3905640, 79.2939682
6: -62.3606491, 37.4752808, -62.3240204, 37.3703423, -99.7309875, 99.7993011
7: -44.2219658, 40.4108315, -44.0569801, 40.3902893, -84.6122589, 84.4678116
8: -49.8258514, 46.5607376, -49.6510239, 46.5524292, -96.3782654, 96.2117538
9: -40.7670326, 43.8376923, -40.6156883, 43.8088379, -84.5758591, 84.4533844
10: -63.0895081, 58.4032669, -62.9386253, 58.3618469, -121.4513397, 121.3418884
11: -59.3047447, 33.6128464, -59.2510834, 33.5359840, -92.8407135, 92.8639297
12: -60.7295303, 42.3412552, -60.6857643, 42.2378120, -102.9673462, 103.0270233
13: -65.3068695, 61.2479095, -65.1463165, 61.1871986, -126.4940567, 126.3942184
14: -99.2863693, 46.6354942, -99.0973358, 46.6008415, -145.8872070, 145.7328186
15: -47.7599449, 43.0431480, -47.6677666, 42.9742584, -90.7341995, 90.7109070
16: -62.4594841, 45.7399597, -62.3052788, 45.6921196, -108.1516037, 108.0452347
17: -96.2172318, 43.7771454, -96.0392990, 43.7483597, -139.9655914, 139.8164368
18: -59.3502922, 47.8667221, -59.3164520, 47.7664032, -107.1166840, 107.1831741
19: -48.5125084, 27.8759232, -48.4869003, 27.7638168, -76.2763214, 76.3628159
20: -46.4739838, 32.0496063, -46.4546890, 31.9505711, -78.4245529, 78.5042877
21: -58.2010536, 32.6838646, -58.1659050, 32.5798798, -90.7809296, 90.8497696
22: -60.8723831, 34.5526276, -60.8436012, 34.4412460, -95.3136292, 95.3962250
23: -47.0831223, 35.5422592, -47.0475082, 35.4220009, -82.5051270, 82.5897675
24: -57.2939491, 33.9527206, -57.2666702, 33.8541107, -91.1480560, 91.2193909
25: -51.4452133, 37.2894249, -51.4343529, 37.1769104, -88.6221237, 88.7237701
26: -69.9285507, 50.1622047, -69.8622894, 50.0053177, -119.9338684, 120.0244751
27: -56.8609390, 38.8107147, -56.8236313, 38.6876526, -95.5485840, 95.6343460
28: -47.5360069, 39.1883469, -47.5081673, 39.0447121, -86.5807190, 86.6965103
29: -60.0630798, 30.4968510, -60.0280266, 30.3862000, -90.4492798, 90.5248795
30: -58.3698921, 40.4923553, -58.3400726, 40.4291420, -98.7990341, 98.8324280
31: -59.3770676, 34.2944031, -59.3542557, 34.1594086, -93.5364761, 93.6486588
32: -61.0111885, 35.7728233, -60.9794312, 35.6857109, -96.6968994, 96.7522507
33: -86.3747406, 46.3061104, -86.3114166, 46.1917763, -132.5665131, 132.6175232
34: -75.0459671, 31.8685150, -75.0011597, 31.7329865, -106.7789459, 106.8696747
35: -70.8200684, 35.1384201, -70.7775116, 35.0145149, -105.8345795, 105.9159317
36: -71.7366714, 37.7860260, -71.7011871, 37.6383438, -109.3750000, 109.4872131
37: -102.0883789, 33.4077988, -102.0107498, 33.3072433, -135.3956146, 135.4185486
38: -86.8214035, 50.6205750, -86.7685547, 50.4372063, -137.2586060, 137.3891296
39: -97.7099915, 43.9801140, -97.6533279, 43.9211197, -141.6311035, 141.6334381
40: -78.4002457, 34.4317665, -78.3504791, 34.3629456, -112.7631836, 112.7822418
41: -64.5644073, 40.7472191, -64.5084839, 40.6275749, -105.1919708, 105.2557068
42: -48.4956436, 36.0991058, -48.4656181, 35.9981689, -84.4938049, 84.5647278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9704890
time: 99.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0084360
time: 96.72 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -68.3906326, 38.7370682, -68.6696930, 38.8254318, -107.2160645, 107.4067612
1: -36.0820541, 36.4903755, -36.2546463, 36.5775986, -72.6596451, 72.7450256
2: -31.8890991, 38.5291786, -32.0325165, 38.6287613, -70.5178604, 70.5616913
3: -35.4322510, 42.9203568, -35.5659218, 43.0105705, -78.4428101, 78.4862747
4: -41.3392181, 40.2025681, -41.5198212, 40.2792511, -81.6184616, 81.7223816
5: -36.9094276, 42.3301582, -37.0561638, 42.4451447, -79.3545685, 79.3863220
6: -62.3187675, 37.3467331, -62.3905869, 37.4958916, -99.8146591, 99.7373199
7: -44.0511093, 40.3385048, -44.2443314, 40.4740829, -84.5251923, 84.5828400
8: -49.6341209, 46.5086479, -49.8461914, 46.6085052, -96.2426300, 96.3548355
9: -40.6279144, 43.7702408, -40.8031387, 43.8930435, -84.5209503, 84.5733795
10: -62.9293633, 58.3330765, -63.1197433, 58.4561539, -121.3855133, 121.4528198
11: -59.2317924, 33.5709915, -59.3370895, 33.6491547, -92.8809509, 92.9080811
12: -60.6919289, 42.1678772, -60.7594986, 42.3470192, -103.0389328, 102.9273758
13: -65.2113953, 61.1934891, -65.3732147, 61.3129539, -126.5243301, 126.5667038
14: -99.1270828, 46.5718727, -99.3544083, 46.6849861, -145.8120728, 145.9262695
15: -47.6590805, 42.9851952, -47.7896919, 43.0745697, -90.7336502, 90.7748871
16: -62.3092499, 45.6539841, -62.5086479, 45.8049736, -108.1142120, 108.1626282
17: -96.0870514, 43.7233009, -96.2859955, 43.8148651, -139.9019165, 140.0092926
18: -59.2843819, 47.7424202, -59.3990860, 47.8776512, -107.1620331, 107.1415024
19: -48.4565048, 27.7995090, -48.5782089, 27.9043121, -76.3608170, 76.3777161
20: -46.4240837, 31.9666004, -46.5314941, 32.0784988, -78.5025787, 78.4980927
21: -58.1416931, 32.6098862, -58.2646904, 32.7137375, -90.8554230, 90.8745728
22: -60.7826767, 34.4504852, -60.9701271, 34.5737801, -95.3564606, 95.4206085
23: -47.0369873, 35.4773598, -47.1398048, 35.5851517, -82.6221390, 82.6171570
24: -57.2444725, 33.8929787, -57.3704567, 33.9836235, -91.2280884, 91.2634354
25: -51.3733673, 37.1932220, -51.5251427, 37.3159332, -88.6892853, 88.7183533
26: -69.8361664, 50.0205612, -70.0079880, 50.1936226, -120.0297852, 120.0285492
27: -56.8041458, 38.7353973, -56.9397430, 38.8494263, -95.6535645, 95.6751404
28: -47.4753342, 39.0873680, -47.6121330, 39.2272186, -86.7025452, 86.6994934
29: -59.9858818, 30.4218731, -60.1504631, 30.5287304, -90.5146103, 90.5723343
30: -58.3038254, 40.4484100, -58.4069405, 40.5278397, -98.8316650, 98.8553467
31: -59.3081665, 34.1747284, -59.4551582, 34.3195953, -93.6277466, 93.6298828
32: -60.9636116, 35.6533127, -61.0512047, 35.7803802, -96.7439880, 96.7045135
33: -86.2876282, 46.1502647, -86.4354324, 46.3142586, -132.6018829, 132.5856781
34: -74.9541321, 31.7182369, -75.1170959, 31.8859921, -106.8401260, 106.8353348
35: -70.7257385, 34.9773102, -70.8960648, 35.1405792, -105.8663177, 105.8733673
36: -71.6372833, 37.6000404, -71.8192596, 37.7869492, -109.4242325, 109.4192963
37: -101.9810638, 33.2716827, -102.1567154, 33.4098053, -135.3908691, 135.4284058
38: -86.6792297, 50.3759422, -86.9254837, 50.6272316, -137.3064575, 137.3014221
39: -97.6250153, 43.8466835, -97.7878265, 43.9618988, -141.5869141, 141.6345062
40: -78.3399658, 34.3031082, -78.4485550, 34.4248123, -112.7647781, 112.7516632
41: -64.5025558, 40.6501465, -64.6097794, 40.7775955, -105.2801361, 105.2599258
42: -48.4545555, 36.0103912, -48.5233765, 36.1315613, -84.5861206, 84.5337677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0141910, upper bound: 50.9802215
time: 195.21 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0141910, upper bound: 50.9802215
time: 742.72 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -68.8513336, 38.8107414, -68.8868027, 38.8475876, -107.6989059, 107.6975327
1: -36.3815422, 36.5468941, -36.3973236, 36.5874634, -72.9689941, 72.9442139
2: -32.1409836, 38.5936279, -32.1532364, 38.6404800, -70.7814560, 70.7468567
3: -35.6679840, 42.9916115, -35.6790161, 43.0300789, -78.6980591, 78.6706238
4: -41.6535645, 40.2581635, -41.6697044, 40.2945633, -81.9481201, 81.9278641
5: -37.1662598, 42.4113007, -37.1791382, 42.4634094, -79.6296692, 79.5904388
6: -62.3993263, 37.5907135, -62.4191322, 37.6106300, -100.0099487, 100.0098419
7: -44.3925781, 40.4272804, -44.4086647, 40.4891434, -84.8817215, 84.8359451
8: -50.0041008, 46.5854645, -50.0232697, 46.6297684, -96.6338577, 96.6087265
9: -40.9229202, 43.8581963, -40.9431458, 43.9114494, -84.8343658, 84.8013458
10: -63.2468338, 58.4423065, -63.2697945, 58.4896202, -121.7364502, 121.7120972
11: -59.3720245, 33.6921921, -59.4028282, 33.7055740, -93.0775986, 93.0950165
12: -60.7756729, 42.4540482, -60.7926216, 42.4794731, -103.2551422, 103.2466660
13: -65.4763260, 61.2967949, -65.4978027, 61.3532639, -126.8295746, 126.7946014
14: -99.4869385, 46.6616516, -99.5225067, 46.7070618, -146.1940002, 146.1841583
15: -47.8576775, 43.1211548, -47.8803139, 43.1379089, -90.9955750, 91.0014648
16: -62.6273842, 45.7651520, -62.6585770, 45.8278923, -108.4552765, 108.4237213
17: -96.4043732, 43.8130188, -96.4329834, 43.8479424, -140.2523193, 140.2460022
18: -59.3953934, 47.9718056, -59.4380493, 47.9872284, -107.3826141, 107.4098511
19: -48.5454559, 27.9907036, -48.6057014, 27.9973183, -76.5427704, 76.5964050
20: -46.5032196, 32.1557045, -46.5548668, 32.1688385, -78.6720581, 78.7105713
21: -58.2522736, 32.7913246, -58.3076286, 32.8007050, -91.0529785, 91.0989456
22: -60.9147263, 34.6692314, -61.0065651, 34.6794357, -95.5941544, 95.6757965
23: -47.1125603, 35.6641083, -47.1643181, 35.6743011, -82.7868500, 82.8284302
24: -57.3359070, 34.0522423, -57.4027481, 34.0599480, -91.3958588, 91.4549866
25: -51.4772301, 37.4064903, -51.5529480, 37.4182816, -88.8955078, 88.9594421
26: -69.9779663, 50.3264275, -70.0446854, 50.3409233, -120.3188934, 120.3711090
27: -56.9056854, 38.9353104, -56.9760399, 38.9445343, -95.8502197, 95.9113464
28: -47.5612640, 39.3345032, -47.6323776, 39.3451653, -86.9064331, 86.9668732
29: -60.1117973, 30.6095848, -60.1925621, 30.6187782, -90.7305756, 90.8021469
30: -58.4204788, 40.5623283, -58.4582253, 40.5795288, -99.0000000, 99.0205383
31: -59.4206734, 34.4322281, -59.4933548, 34.4436493, -93.8643188, 93.9255829
32: -61.0529251, 35.8663712, -61.0831223, 35.8804245, -96.9333420, 96.9494858
33: -86.4335098, 46.4223862, -86.4805450, 46.4429970, -132.8765106, 132.9029236
34: -75.0801544, 32.0115814, -75.1452866, 32.0265770, -107.1067352, 107.1568680
35: -70.8609772, 35.2646561, -70.9282074, 35.2784805, -106.1394577, 106.1928635
36: -71.7685318, 37.9356079, -71.8449478, 37.9483719, -109.7169037, 109.7805481
37: -102.1476517, 33.5136452, -102.2040787, 33.5260620, -135.6737061, 135.7177124
38: -86.8625183, 50.8136520, -86.9616241, 50.8367195, -137.6992188, 137.7752686
39: -97.7811737, 44.0402527, -97.8391037, 44.0539856, -141.8351593, 141.8793640
40: -78.4592896, 34.5080261, -78.4957886, 34.5207253, -112.9800110, 113.0037994
41: -64.6066666, 40.8711395, -64.6408234, 40.8823395, -105.4889832, 105.5119629
42: -48.5253868, 36.2080803, -48.5461693, 36.2249298, -84.7503204, 84.7542496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0569152, upper bound: 50.9857949
time: 97.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0237466
time: 94.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -68.4649506, 38.7987747, -68.1813965, 38.7610817, -107.2260056, 106.9801636
1: -36.1451607, 36.5656204, -35.9443512, 36.5451622, -72.6903229, 72.5099716
2: -31.9329414, 38.6189346, -31.7701454, 38.5968552, -70.5298004, 70.3890839
3: -35.4638901, 42.9918327, -35.3252487, 42.9591484, -78.4230347, 78.3170776
4: -41.3743439, 40.2695618, -41.1706467, 40.2410889, -81.6154327, 81.4402084
5: -36.9234886, 42.4277649, -36.7903862, 42.3839340, -79.3074188, 79.2181473
6: -62.3502502, 37.3587875, -62.3111343, 37.2705612, -99.6208115, 99.6699142
7: -44.1065445, 40.4520416, -43.9016571, 40.4297676, -84.5363007, 84.3536987
8: -49.7195892, 46.5945740, -49.4884453, 46.5704041, -96.2899933, 96.0830078
9: -40.6675491, 43.8695831, -40.4909286, 43.8394203, -84.5069580, 84.3605042
10: -62.9876633, 58.4151878, -62.8055916, 58.3690948, -121.3567581, 121.2207794
11: -59.3044052, 33.5798149, -59.2120056, 33.4908447, -92.7952423, 92.7918243
12: -60.7270622, 42.1991425, -60.6676407, 42.1262932, -102.8533325, 102.8667831
13: -65.2098083, 61.2992210, -65.0358353, 61.1945076, -126.4043121, 126.3350449
14: -99.1452713, 46.6558266, -98.9523468, 46.6170044, -145.7622681, 145.6081696
15: -47.7089577, 42.9967270, -47.5961494, 42.9239159, -90.6328735, 90.5928726
16: -62.3851929, 45.7691727, -62.1781616, 45.7262993, -108.1114883, 107.9473343
17: -96.1393204, 43.7884026, -95.9094009, 43.7435341, -139.8828583, 139.6977997
18: -59.3698654, 47.7907906, -59.3144722, 47.6682281, -107.0380936, 107.1052628
19: -48.5738297, 27.8227177, -48.5125465, 27.6759148, -76.2497406, 76.3352661
20: -46.5163040, 31.9980412, -46.4770164, 31.8688393, -78.3851471, 78.4750595
21: -58.2500496, 32.6335449, -58.1711998, 32.5003662, -90.7504120, 90.8047485
22: -60.9576187, 34.4978714, -60.8880806, 34.3440247, -95.3016434, 95.3859482
23: -47.1352043, 35.4805069, -47.0684967, 35.3412781, -82.4764862, 82.5490036
24: -57.3628426, 33.9115067, -57.2942696, 33.7844620, -91.1473083, 91.2057724
25: -51.5149002, 37.2504768, -51.4722023, 37.0841980, -88.5990906, 88.7226791
26: -69.9565964, 50.0376472, -69.8863373, 49.8661232, -119.8227234, 119.9239807
27: -56.9364700, 38.7547264, -56.8495445, 38.6003342, -95.5368042, 95.6042709
28: -47.6069489, 39.1195374, -47.5516243, 38.9350433, -86.5419922, 86.6711578
29: -60.1439667, 30.4521446, -60.0561371, 30.3039322, -90.4478989, 90.5082779
30: -58.3604851, 40.4842224, -58.3183784, 40.3916321, -98.7521210, 98.8025970
31: -59.4493790, 34.2279205, -59.3810120, 34.0441971, -93.4935760, 93.6089325
32: -61.0119286, 35.6721497, -60.9706459, 35.5976448, -96.6095734, 96.6427917
33: -86.3694992, 46.1754646, -86.3042984, 46.0771675, -132.4466553, 132.4797668
34: -75.0752106, 31.7548027, -75.0317841, 31.6009541, -106.6761627, 106.7865906
35: -70.8488235, 35.0270309, -70.8058167, 34.8850136, -105.7338257, 105.8328476
36: -71.7792664, 37.6634979, -71.7452240, 37.4841766, -109.2634430, 109.4087143
37: -102.0876923, 33.2419281, -102.0114288, 33.1993256, -135.2870178, 135.2533569
38: -86.8653030, 50.4722252, -86.8234558, 50.2422752, -137.1075745, 137.2956848
39: -97.7261581, 43.8899841, -97.6484833, 43.8385849, -141.5647430, 141.5384674
40: -78.4234924, 34.3053741, -78.3344650, 34.2742691, -112.6977615, 112.6398392
41: -64.5687332, 40.6160355, -64.5067291, 40.5308571, -105.0995941, 105.1227646
42: -48.4933968, 35.9806633, -48.4599266, 35.9152832, -84.4086761, 84.4405899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1653

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9806372
time: 114.93 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0185907
time: 107.00 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -68.9235535, 38.8722153, -68.3971786, 38.7832794, -107.7068176, 107.2693939
1: -36.4432373, 36.6220512, -36.0865479, 36.5549850, -72.9982147, 72.7086029
2: -32.1837006, 38.6831055, -31.8902874, 38.6085014, -70.7922058, 70.5733795
3: -35.6983566, 43.0630722, -35.4376526, 42.9785233, -78.6768799, 78.5007248
4: -41.6871567, 40.3246078, -41.3196831, 40.2562180, -81.9433746, 81.6442871
5: -37.1790695, 42.5087814, -36.9127808, 42.4021721, -79.5812378, 79.4215622
6: -62.4286118, 37.6016769, -62.3389320, 37.3845291, -99.8131409, 99.9406128
7: -44.4466743, 40.5406075, -44.0655937, 40.4448242, -84.8914948, 84.6062012
8: -50.0874596, 46.6710968, -49.6643410, 46.5915031, -96.6789627, 96.3354340
9: -40.9612503, 43.9571228, -40.6302185, 43.8577728, -84.8190231, 84.5873337
10: -63.3028564, 58.5239029, -62.9546051, 58.4025078, -121.7053604, 121.4785080
11: -59.4439011, 33.6995239, -59.2758522, 33.5463295, -92.9902115, 92.9753723
12: -60.8095016, 42.4841766, -60.6996384, 42.2574654, -103.0669708, 103.1838074
13: -65.4725037, 61.4011688, -65.1593857, 61.2347832, -126.7072906, 126.5605545
14: -99.5035172, 46.7445526, -99.1201553, 46.6391563, -146.1426697, 145.8647003
15: -47.9062538, 43.1316338, -47.6857948, 42.9866333, -90.8928833, 90.8174286
16: -62.7019119, 45.8799133, -62.3276253, 45.7492981, -108.4512100, 108.2075348
17: -96.4549255, 43.8771439, -96.0581055, 43.7762909, -140.2312164, 139.9352417
18: -59.4806709, 48.0177460, -59.3534012, 47.7766342, -107.2572937, 107.3711395
19: -48.6624069, 28.0117054, -48.5400505, 27.7679443, -76.4303513, 76.5517502
20: -46.5948868, 32.1855812, -46.5002174, 31.9583702, -78.5532532, 78.6857986
21: -58.3597336, 32.8126831, -58.2141151, 32.5862427, -90.9459686, 91.0267944
22: -61.0890350, 34.7146149, -60.9247360, 34.4487114, -95.5377426, 95.6393509
23: -47.2105293, 35.6650429, -47.0928879, 35.4293900, -82.6399231, 82.7579346
24: -57.4539871, 34.0681763, -57.3260117, 33.8593826, -91.3133545, 91.3941879
25: -51.6183281, 37.4615173, -51.4998360, 37.1853790, -88.8036957, 88.9613495
26: -70.0981598, 50.3411674, -69.9229050, 50.0123978, -120.1105576, 120.2640610
27: -57.0368423, 38.9520493, -56.8857765, 38.6940498, -95.7308884, 95.8378296
28: -47.6924019, 39.3643494, -47.5717125, 39.0518036, -86.7442017, 86.9360580
29: -60.2688751, 30.6381798, -60.0984268, 30.3931122, -90.6619873, 90.7365952
30: -58.4772911, 40.5954704, -58.3699493, 40.4416656, -98.9189529, 98.9654236
31: -59.5614624, 34.4829521, -59.4188995, 34.1670418, -93.7285004, 93.9018555
32: -61.1008072, 35.8846893, -61.0023308, 35.6968384, -96.7976379, 96.8870087
33: -86.5148697, 46.4463158, -86.3495331, 46.2053452, -132.7202148, 132.7958374
34: -75.2005615, 32.0472450, -75.0598145, 31.7413712, -106.9419250, 107.1070557
35: -70.9838562, 35.3128281, -70.8379822, 35.0225143, -106.0063477, 106.1508102
36: -71.9103546, 37.9979134, -71.7710342, 37.6450577, -109.5554123, 109.7689514
37: -102.2533951, 33.4827271, -102.0586014, 33.3149338, -135.5683289, 135.5413208
38: -87.0480042, 50.9085274, -86.8595657, 50.4513092, -137.4993134, 137.7680969
39: -97.8825073, 44.0825195, -97.7002945, 43.9305878, -141.8130798, 141.7828064
40: -78.5430069, 34.5088463, -78.3809128, 34.3710060, -112.9140167, 112.8897476
41: -64.6715393, 40.8359909, -64.5375977, 40.6348457, -105.3063812, 105.3735809
42: -48.5629616, 36.1773224, -48.4826050, 36.0081291, -84.5710907, 84.6599274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9863094
time: 88.43 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0242543
time: 100.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -68.7077103, 38.8273544, -68.6921844, 38.8565941, -107.5642853, 107.5195389
1: -36.3001671, 36.5785522, -36.2647476, 36.6135902, -72.9137573, 72.8432999
2: -32.0642014, 38.6331291, -32.0393219, 38.6705322, -70.7347336, 70.6724548
3: -35.5835724, 43.0141106, -35.5721588, 43.0443916, -78.6279602, 78.5862732
4: -41.5469131, 40.2873917, -41.5287666, 40.3110733, -81.8579788, 81.8161545
5: -37.0544357, 42.4507713, -37.0618858, 42.4910278, -79.5454636, 79.5126495
6: -62.3886070, 37.4699326, -62.4048271, 37.5083809, -99.8969879, 99.8747559
7: -44.2779198, 40.4683952, -44.2525139, 40.5287399, -84.8066559, 84.7209091
8: -49.8976364, 46.6191750, -49.8584785, 46.6472664, -96.5448837, 96.4776535
9: -40.8207703, 43.8901596, -40.8163834, 43.9403763, -84.7611389, 84.7065353
10: -63.1430702, 58.4538345, -63.1342812, 58.4953423, -121.6384125, 121.5881195
11: -59.3718185, 33.6594925, -59.3622742, 33.6575470, -93.0293655, 93.0217667
12: -60.7739410, 42.3074265, -60.7722435, 42.3654060, -103.1393433, 103.0796661
13: -65.3762207, 61.3474045, -65.3834381, 61.3625832, -126.7388000, 126.7308426
14: -99.3441696, 46.6816177, -99.3748016, 46.7234802, -146.0676422, 146.0564270
15: -47.8051834, 43.0742798, -47.8059692, 43.0866776, -90.8918610, 90.8802490
16: -62.5514793, 45.7943268, -62.5308418, 45.8607941, -108.4122696, 108.3251648
17: -96.3242569, 43.8237915, -96.3029099, 43.8432426, -140.1674805, 140.1266937
18: -59.4146614, 47.8944130, -59.4356232, 47.8867950, -107.3014526, 107.3300323
19: -48.6064453, 27.9368725, -48.6311874, 27.9072857, -76.5137329, 76.5680542
20: -46.5450020, 32.1033859, -46.5764046, 32.0849571, -78.6299591, 78.6797943
21: -58.3006096, 32.7400665, -58.3125572, 32.7185516, -91.0191650, 91.0526199
22: -60.9996147, 34.6145325, -61.0507851, 34.5807457, -95.5803528, 95.6653061
23: -47.1645470, 35.6011658, -47.1852760, 35.5910187, -82.7555695, 82.7864380
24: -57.4047279, 34.0096321, -57.4294739, 33.9875221, -91.3922501, 91.4390945
25: -51.5464706, 37.3671379, -51.5923309, 37.3235588, -88.8700256, 88.9594650
26: -70.0050201, 50.1989861, -70.0667953, 50.1994438, -120.2044678, 120.2657776
27: -56.9804573, 38.8771706, -57.0015869, 38.8537521, -95.8341980, 95.8787537
28: -47.6316338, 39.2645607, -47.6752777, 39.2327423, -86.8643723, 86.9398346
29: -60.1929131, 30.5642891, -60.2208824, 30.5348511, -90.7277679, 90.7851639
30: -58.4100456, 40.5522003, -58.4359589, 40.5386009, -98.9486465, 98.9881592
31: -59.4927826, 34.3651886, -59.5193748, 34.3260803, -93.8188629, 93.8845673
32: -61.0538292, 35.7659187, -61.0736809, 35.7903519, -96.8441620, 96.8395920
33: -86.4276886, 46.2905273, -86.4736786, 46.3273125, -132.7550049, 132.7642059
34: -75.1091003, 31.8948708, -75.1721954, 31.8939514, -107.0030365, 107.0670624
35: -70.8892822, 35.1522827, -70.9556732, 35.1479073, -106.0371857, 106.1079559
36: -71.8108215, 37.8113556, -71.8853760, 37.7929840, -109.6038055, 109.6967316
37: -102.1467361, 33.3494339, -102.2029114, 33.4170647, -135.5637970, 135.5523376
38: -86.9060974, 50.6604462, -87.0100708, 50.6403961, -137.5464935, 137.6705170
39: -97.7971649, 43.9508629, -97.8340683, 43.9711342, -141.7682953, 141.7849274
40: -78.4832306, 34.3801270, -78.4780426, 34.4317093, -112.9149399, 112.8581696
41: -64.6107788, 40.7375069, -64.6381912, 40.7830238, -105.3937988, 105.3756943
42: -48.5236969, 36.0889206, -48.5399857, 36.1391068, -84.6628036, 84.6289062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0167462
time: 102.07 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0546244
time: 90.51 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -69.1668472, 38.9006882, -68.9093781, 38.8787766, -108.0456238, 107.8100586
1: -36.5985031, 36.6349487, -36.4074821, 36.6234207, -73.2219238, 73.0424271
2: -32.3153038, 38.6972885, -32.1600800, 38.6822205, -70.9975281, 70.8573685
3: -35.8184662, 43.0852470, -35.6851959, 43.0639305, -78.8823853, 78.7704315
4: -41.8604889, 40.3424454, -41.6787872, 40.3263168, -82.1868057, 82.0212326
5: -37.3104095, 42.5317078, -37.1848373, 42.5092850, -79.8196945, 79.7165451
6: -62.4676437, 37.7136345, -62.4335213, 37.6232338, -100.0908661, 100.1471558
7: -44.6183243, 40.5568924, -44.4168243, 40.5437889, -85.1621094, 84.9737167
8: -50.2664146, 46.6955872, -50.0355606, 46.6684875, -96.9348907, 96.7311478
9: -41.1150246, 43.9776230, -40.9564781, 43.9587593, -85.0737839, 84.9340973
10: -63.4592056, 58.5624352, -63.2843323, 58.5287285, -121.9879303, 121.8467636
11: -59.5119438, 33.7800560, -59.4278946, 33.7139549, -93.2258987, 93.2079391
12: -60.8569946, 42.5931473, -60.8054237, 42.4977493, -103.3547440, 103.3985672
13: -65.6398315, 61.4493446, -65.5083008, 61.4027901, -127.0426178, 126.9576416
14: -99.7028198, 46.7702217, -99.5430069, 46.7454834, -146.4483032, 146.3132172
15: -48.0031624, 43.2095566, -47.8965912, 43.1499748, -91.1531372, 91.1061478
16: -62.8682060, 45.9050217, -62.6806984, 45.8837547, -108.7519608, 108.5857162
17: -96.6390762, 43.9128304, -96.4499283, 43.8762283, -140.5153046, 140.3627319
18: -59.5249748, 48.1224136, -59.4743996, 47.9963913, -107.5213623, 107.5968170
19: -48.6949081, 28.1267052, -48.6586151, 28.0003033, -76.6952133, 76.7853165
20: -46.6235924, 32.2913895, -46.5996933, 32.1752930, -78.7988892, 78.8910828
21: -58.4102478, 32.9201508, -58.3553505, 32.8055573, -91.2157822, 91.2754974
22: -61.1308060, 34.8320274, -61.0871925, 34.6863899, -95.8171997, 95.9192200
23: -47.2395973, 35.7866592, -47.2096825, 35.6802216, -82.9198151, 82.9963379
24: -57.4955521, 34.1674728, -57.4616241, 34.0638351, -91.5593872, 91.6290970
25: -51.6497955, 37.5790787, -51.6200790, 37.4258957, -89.0756912, 89.1991425
26: -70.1464767, 50.5031548, -70.1033478, 50.3467102, -120.4931870, 120.6064987
27: -57.0808105, 39.0757599, -57.0376701, 38.9489517, -96.0297546, 96.1134338
28: -47.7170448, 39.5103188, -47.6954079, 39.3506851, -87.0677185, 87.2057266
29: -60.3175240, 30.7511101, -60.2629204, 30.6249428, -90.9424515, 91.0140305
30: -58.5268211, 40.6652298, -58.4876022, 40.5903549, -99.1171722, 99.1528244
31: -59.6047401, 34.6212311, -59.5575523, 34.4501610, -94.0549011, 94.1787872
32: -61.1425056, 35.9786949, -61.1056366, 35.8902893, -97.0327911, 97.0843201
33: -86.5728455, 46.5616722, -86.5187836, 46.4560928, -133.0289307, 133.0804596
34: -75.2342529, 32.1873055, -75.2003174, 32.0345383, -107.2687836, 107.3876190
35: -71.0239639, 35.4384995, -70.9878082, 35.2858582, -106.3098221, 106.4263077
36: -71.9416428, 38.1460876, -71.9110870, 37.9544067, -109.8960495, 110.0571671
37: -102.3121643, 33.5906830, -102.2501984, 33.5333481, -135.8455048, 135.8408813
38: -87.0886230, 51.0968819, -87.0462265, 50.8498688, -137.9384766, 138.1430969
39: -97.9527435, 44.1433296, -97.8855896, 44.0631523, -142.0158844, 142.0289154
40: -78.6029663, 34.5841217, -78.5252991, 34.5276947, -113.1306610, 113.1094208
41: -64.7133484, 40.9579659, -64.6690521, 40.8878098, -105.6011581, 105.6270142
42: -48.5933189, 36.2857971, -48.5628052, 36.2325516, -84.8258667, 84.8486023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=436, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0221640
time: 115.01 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0600534
time: 115.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 233.20 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9646235
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0025596
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9704890
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0084360
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141910, upper bound: 50.9802215
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141910, upper bound: 50.9802215
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0569152, upper bound: 50.9857949
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0237466
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9806372
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0185907
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9863094
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0242543
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0167462
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0546244
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0221640
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.20
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0600534

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -68.5978851, 38.7807541, -68.3554611, 38.7479477, -107.3458328, 107.1362152
1: -36.2220154, 36.5328903, -36.0634155, 36.5165176, -72.7385330, 72.5962982
2: -32.0051956, 38.5783310, -31.8711739, 38.5641022, -70.5692978, 70.4495087
3: -35.5442543, 42.9677887, -35.4209061, 42.9407578, -78.4850159, 78.3886871
4: -41.4748268, 40.2390213, -41.2960892, 40.2214890, -81.6963120, 81.5351105
5: -37.0306091, 42.3866806, -36.8961067, 42.3527985, -79.3834076, 79.2827835
6: -62.3585358, 37.4688454, -62.3188133, 37.3547592, -99.7132950, 99.7876587
7: -44.2175293, 40.4099579, -44.0460854, 40.3881378, -84.6056671, 84.4560394
8: -49.8198166, 46.5589294, -49.6362534, 46.5480804, -96.3678894, 96.1951828
9: -40.7621117, 43.8365517, -40.6043205, 43.8060303, -84.5681458, 84.4408646
10: -63.0830345, 58.4014168, -62.9231949, 58.3575401, -121.4405670, 121.3245926
11: -59.3022461, 33.6087837, -59.2449799, 33.5270653, -92.8293152, 92.8537521
12: -60.7270432, 42.3371582, -60.6801414, 42.2278061, -102.9548340, 103.0173035
13: -65.3007965, 61.2452965, -65.1334534, 61.1808891, -126.4816895, 126.3787537
14: -99.2792130, 46.6338425, -99.0806885, 46.5967941, -145.8760071, 145.7145386
15: -47.7544136, 43.0411758, -47.6542320, 42.9694023, -90.7237930, 90.6954041
16: -62.4546432, 45.7385864, -62.2936058, 45.6887817, -108.1434250, 108.0321960
17: -96.2107925, 43.7749176, -96.0238800, 43.7430038, -139.9537964, 139.7987976
18: -59.3479080, 47.8618774, -59.3106575, 47.7545319, -107.1024323, 107.1725311
19: -48.5105095, 27.8711491, -48.4820061, 27.7520561, -76.2625656, 76.3531570
20: -46.4725800, 32.0441971, -46.4512253, 31.9373894, -78.4099579, 78.4954224
21: -58.1986771, 32.6777992, -58.1601143, 32.5650215, -90.7636871, 90.8379135
22: -60.8703270, 34.5479546, -60.8386040, 34.4301300, -95.3004608, 95.3865585
23: -47.0810127, 35.5378265, -47.0423927, 35.4111633, -82.4921722, 82.5802155
24: -57.2915878, 33.9487152, -57.2608795, 33.8443871, -91.1359711, 91.2095947
25: -51.4432182, 37.2846031, -51.4294395, 37.1650696, -88.6082764, 88.7140350
26: -69.9267426, 50.1554260, -69.8579636, 49.9886932, -119.9154358, 120.0133896
27: -56.8590317, 38.8054161, -56.8189774, 38.6746445, -95.5336761, 95.6243820
28: -47.5345306, 39.1829300, -47.5045967, 39.0314484, -86.5659714, 86.6875229
29: -60.0607491, 30.4929314, -60.0222931, 30.3766232, -90.4373703, 90.5152206
30: -58.3677101, 40.4869461, -58.3347282, 40.4160423, -98.7837524, 98.8216705
31: -59.3742180, 34.2888718, -59.3473091, 34.1458282, -93.5200500, 93.6361847
32: -61.0089378, 35.7685394, -60.9739304, 35.6752396, -96.6841736, 96.7424622
33: -86.3718567, 46.3011169, -86.3046265, 46.1794205, -132.5512695, 132.6057434
34: -75.0437775, 31.8636684, -74.9958801, 31.7211838, -106.7649460, 106.8595428
35: -70.8181305, 35.1341820, -70.7728348, 35.0041428, -105.8222733, 105.9070129
36: -71.7353210, 37.7808228, -71.6979675, 37.6256332, -109.3609543, 109.4787903
37: -102.0853653, 33.4044571, -102.0035629, 33.2989960, -135.3843689, 135.4080200
38: -86.8195496, 50.6142845, -86.7640991, 50.4218521, -137.2413940, 137.3783875
39: -97.7067642, 43.9729309, -97.6455841, 43.9056396, -141.6123810, 141.6185150
40: -78.3973541, 34.4263611, -78.3436584, 34.3497353, -112.7470856, 112.7700195
41: -64.5622101, 40.7426414, -64.5032349, 40.6164284, -105.1786346, 105.2458801
42: -48.4937286, 36.0953064, -48.4610214, 35.9896774, -84.4834061, 84.5563278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9771141
time: 95.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0042155
time: 102.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -68.6428528, 38.7830811, -68.4466858, 38.7294197, -107.3722687, 107.2297592
1: -36.2578392, 36.5319824, -36.1390381, 36.5046234, -72.7624664, 72.6710205
2: -32.0216713, 38.5771408, -31.9062481, 38.5557823, -70.5774536, 70.4833908
3: -35.5586014, 42.9608498, -35.4518166, 42.9119034, -78.4704971, 78.4126663
4: -41.5147896, 40.2351227, -41.3798904, 40.2139435, -81.7287292, 81.6150131
5: -37.0575333, 42.3818283, -36.9529762, 42.3469238, -79.4044571, 79.3348083
6: -62.3645401, 37.4134521, -62.2959023, 37.2395287, -99.6040649, 99.7093506
7: -44.2828751, 40.4096146, -44.1753426, 40.4057121, -84.6885834, 84.5849609
8: -49.8427582, 46.5619507, -49.6870041, 46.5267944, -96.3695450, 96.2489319
9: -40.7945709, 43.8333206, -40.6752205, 43.8154182, -84.6099854, 84.5085373
10: -63.0979042, 58.3964615, -62.9581451, 58.3485146, -121.4464188, 121.3546066
11: -59.3314133, 33.5846977, -59.2479935, 33.4808540, -92.8122711, 92.8326874
12: -60.7199974, 42.3732834, -60.6669388, 42.2844467, -103.0044403, 103.0402222
13: -65.3183899, 61.2468567, -65.1630630, 61.1816635, -126.5000534, 126.4099121
14: -99.2612305, 46.6260033, -99.0386429, 46.5586662, -145.8198853, 145.6646423
15: -47.7079849, 43.0757332, -47.5657005, 43.0160789, -90.7240601, 90.6414337
16: -62.5074005, 45.7329750, -62.3953667, 45.7123260, -108.2197266, 108.1283340
17: -96.2522430, 43.7757187, -96.0962524, 43.6771011, -139.9293518, 139.8719788
18: -59.3545532, 47.8507500, -59.3077965, 47.7337494, -107.0882950, 107.1585464
19: -48.5069923, 27.8599319, -48.4400635, 27.7298985, -76.2368927, 76.2999954
20: -46.4688835, 32.0133743, -46.3877449, 31.8761120, -78.3449936, 78.4011230
21: -58.2027054, 32.6277084, -58.1083832, 32.4661865, -90.6688919, 90.7360916
22: -60.8785553, 34.5519791, -60.8231354, 34.4366684, -95.3152161, 95.3751068
23: -47.0772438, 35.5551682, -47.0203590, 35.4502106, -82.5274506, 82.5755234
24: -57.3013268, 33.9476089, -57.2246094, 33.8461685, -91.1474838, 91.1722031
25: -51.4409599, 37.2823029, -51.4006271, 37.1595154, -88.6004791, 88.6829300
26: -69.9375458, 50.1584396, -69.8631821, 49.9988708, -119.9364166, 120.0216217
27: -56.8658028, 38.7936363, -56.7900200, 38.6545067, -95.5203094, 95.5836487
28: -47.5307503, 39.1955032, -47.4856186, 39.0627213, -86.5934753, 86.6811218
29: -60.0760231, 30.5144730, -60.0143280, 30.4198761, -90.4958954, 90.5288010
30: -58.3783302, 40.4268951, -58.2770348, 40.2978096, -98.6761169, 98.7039337
31: -59.3765030, 34.2906113, -59.3261375, 34.1523438, -93.5288467, 93.6167450
32: -61.0092621, 35.7503395, -60.9513893, 35.6249847, -96.6342316, 96.7017288
33: -86.3743134, 46.2945366, -86.3024445, 46.1722031, -132.5465088, 132.5969696
34: -75.0419769, 31.8882408, -75.0210266, 31.7712955, -106.8132629, 106.9092712
35: -70.8255005, 35.1626434, -70.8047714, 35.0634155, -105.8889160, 105.9674149
36: -71.7392273, 37.8027229, -71.7075653, 37.6721039, -109.4113159, 109.5102844
37: -102.0843658, 33.4276810, -101.9828339, 33.3432465, -135.4276123, 135.4105072
38: -86.8207016, 50.6576767, -86.7864838, 50.5051765, -137.3258667, 137.4441528
39: -97.7101746, 43.9659653, -97.6228714, 43.8871803, -141.5973511, 141.5888214
40: -78.4053650, 34.4502373, -78.3459702, 34.3911438, -112.7965012, 112.7962036
41: -64.5673218, 40.7396545, -64.5108948, 40.6073151, -105.1746368, 105.2505493
42: -48.4881172, 36.1060181, -48.4584961, 36.0058479, -84.4939651, 84.5645142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9558195
time: 108.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9817063
time: 93.24 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -68.8437576, 38.8095970, -68.8690796, 38.8447418, -107.6884995, 107.6786804
1: -36.3763924, 36.5461006, -36.3846588, 36.5855026, -72.9618988, 72.9307556
2: -32.1362686, 38.5927124, -32.1416473, 38.6383286, -70.7745972, 70.7343597
3: -35.6639633, 42.9900742, -35.6691704, 43.0263786, -78.6903381, 78.6592407
4: -41.6482849, 40.2568512, -41.6568298, 40.2913589, -81.9396439, 81.9136810
5: -37.1622696, 42.4099731, -37.1692886, 42.4602814, -79.6225433, 79.5792618
6: -62.3971786, 37.5843010, -62.4139328, 37.5950432, -99.9922028, 99.9982300
7: -44.3881378, 40.4264030, -44.3977890, 40.4869881, -84.8751221, 84.8241882
8: -49.9981003, 46.5836639, -50.0084648, 46.6254044, -96.6235046, 96.5921326
9: -40.9179955, 43.8570480, -40.9317627, 43.9086494, -84.8266296, 84.7888107
10: -63.2404137, 58.4405212, -63.2543259, 58.4852638, -121.7256775, 121.6948471
11: -59.3695107, 33.6881523, -59.3967514, 33.6966476, -93.0661621, 93.0848999
12: -60.7731628, 42.4499855, -60.7870102, 42.4693871, -103.2425537, 103.2369843
13: -65.4702606, 61.2942123, -65.4849396, 61.3469963, -126.8172379, 126.7791443
14: -99.4798050, 46.6599426, -99.5058823, 46.7029648, -146.1827698, 146.1658325
15: -47.8521461, 43.1191940, -47.8667641, 43.1330490, -90.9851990, 90.9859543
16: -62.6225510, 45.7638092, -62.6468811, 45.8245735, -108.4471207, 108.4106903
17: -96.3979568, 43.8108368, -96.4175491, 43.8425903, -140.2405396, 140.2283936
18: -59.3929749, 47.9670029, -59.4322090, 47.9753799, -107.3683472, 107.3992081
19: -48.5434456, 27.9859161, -48.6008072, 27.9855766, -76.5290222, 76.5867233
20: -46.5018120, 32.1503296, -46.5513954, 32.1556549, -78.6574707, 78.7017212
21: -58.2498894, 32.7852669, -58.3018799, 32.7858658, -91.0357513, 91.0871353
22: -60.9126434, 34.6645889, -61.0016136, 34.6683197, -95.5809479, 95.6661987
23: -47.1104584, 35.6597023, -47.1592102, 35.6634674, -82.7739182, 82.8189087
24: -57.3335648, 34.0482597, -57.3969498, 34.0502205, -91.3837738, 91.4451981
25: -51.4752121, 37.4016571, -51.5480499, 37.4064369, -88.8816528, 88.9497070
26: -69.9761963, 50.3196602, -70.0403824, 50.3242760, -120.3004608, 120.3600388
27: -56.9037437, 38.9299965, -56.9713936, 38.9315338, -95.8352814, 95.9013901
28: -47.5597801, 39.3291054, -47.6287956, 39.3319244, -86.8916931, 86.9579010
29: -60.1094475, 30.6056633, -60.1868553, 30.6091957, -90.7186279, 90.7925186
30: -58.4182816, 40.5569382, -58.4528542, 40.5664215, -98.9847031, 99.0097885
31: -59.4178467, 34.4266815, -59.4864197, 34.4300575, -93.8479004, 93.9130936
32: -61.0506401, 35.8620682, -61.0776215, 35.8699150, -96.9205551, 96.9396896
33: -86.4306488, 46.4173775, -86.4737778, 46.4306946, -132.8613434, 132.8911438
34: -75.0780029, 32.0067596, -75.1400452, 32.0147629, -107.0927658, 107.1468048
35: -70.8589935, 35.2604446, -70.9235153, 35.2681274, -106.1271210, 106.1839523
36: -71.7671738, 37.9304123, -71.8417206, 37.9356346, -109.7028046, 109.7721329
37: -102.1446228, 33.5102463, -102.1969070, 33.5178375, -135.6624451, 135.7071533
38: -86.8606567, 50.8073616, -86.9571609, 50.8213081, -137.6819458, 137.7645264
39: -97.7779388, 44.0330544, -97.8314438, 44.0384369, -141.8163452, 141.8645020
40: -78.4563980, 34.5026665, -78.4889526, 34.5075607, -112.9639587, 112.9916229
41: -64.6044693, 40.8665771, -64.6355743, 40.8711891, -105.4756622, 105.5021515
42: -48.5234680, 36.2042885, -48.5415535, 36.2164612, -84.7399292, 84.7458420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9558195
time: 129.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9817063
time: 100.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -68.4573517, 38.7976227, -68.1637192, 38.7582397, -107.2155914, 106.9613419
1: -36.1399841, 36.5648193, -35.9316711, 36.5432014, -72.6831741, 72.4964905
2: -31.9281998, 38.6180344, -31.7585659, 38.5946884, -70.5228882, 70.3765945
3: -35.4598999, 42.9902916, -35.3153877, 42.9554367, -78.4153366, 78.3056793
4: -41.3690758, 40.2682495, -41.1577644, 40.2378883, -81.6069489, 81.4260101
5: -36.9194565, 42.4264374, -36.7805214, 42.3807907, -79.3002472, 79.2069550
6: -62.3481216, 37.3523598, -62.3059235, 37.2549515, -99.6030655, 99.6582718
7: -44.1021194, 40.4511566, -43.8907509, 40.4276161, -84.5297241, 84.3419037
8: -49.7135620, 46.5927887, -49.4736671, 46.5660591, -96.2796021, 96.0664520
9: -40.6626358, 43.8684425, -40.4795761, 43.8365936, -84.4992218, 84.3480225
10: -62.9811974, 58.4133797, -62.7901459, 58.3647537, -121.3459473, 121.2035141
11: -59.3019142, 33.5757942, -59.2059174, 33.4819260, -92.7838440, 92.7817078
12: -60.7245789, 42.1950989, -60.6619949, 42.1162720, -102.8408432, 102.8570938
13: -65.2037582, 61.2966232, -65.0229645, 61.1882477, -126.3920059, 126.3195877
14: -99.1381378, 46.6541290, -98.9357224, 46.6129608, -145.7510986, 145.5898438
15: -47.7034340, 42.9947510, -47.5826302, 42.9190521, -90.6224823, 90.5773773
16: -62.3803711, 45.7678146, -62.1664963, 45.7229767, -108.1033401, 107.9343109
17: -96.1329346, 43.7862015, -95.8939133, 43.7381897, -139.8711243, 139.6801147
18: -59.3674583, 47.7859306, -59.3086433, 47.6563721, -107.0238342, 107.0945740
19: -48.5718269, 27.8179264, -48.5076675, 27.6641521, -76.2359619, 76.3255920
20: -46.5148926, 31.9926453, -46.4735527, 31.8556900, -78.3705826, 78.4661865
21: -58.2476921, 32.6275024, -58.1654320, 32.4855194, -90.7332153, 90.7929382
22: -60.9555702, 34.4932022, -60.8830719, 34.3328781, -95.2884369, 95.3762741
23: -47.1331024, 35.4760780, -47.0633774, 35.3304176, -82.4635162, 82.5394592
24: -57.3604774, 33.9075050, -57.2884750, 33.7747307, -91.1351852, 91.1959839
25: -51.5129128, 37.2456322, -51.4672966, 37.0723457, -88.5852509, 88.7129211
26: -69.9548035, 50.0308990, -69.8819885, 49.8494720, -119.8042755, 119.9128876
27: -56.9345360, 38.7494011, -56.8449059, 38.5873260, -95.5218506, 95.5943069
28: -47.6054497, 39.1141243, -47.5480614, 38.9217834, -86.5272369, 86.6621857
29: -60.1416168, 30.4482079, -60.0504227, 30.2943554, -90.4359741, 90.4986191
30: -58.3582726, 40.4788589, -58.3130341, 40.3785248, -98.7368011, 98.7918854
31: -59.4465485, 34.2223816, -59.3740768, 34.0306053, -93.4771576, 93.5964584
32: -61.0096664, 35.6678658, -60.9651260, 35.5871468, -96.5968018, 96.6329880
33: -86.3666077, 46.1704636, -86.2975006, 46.0648766, -132.4314880, 132.4679565
34: -75.0730667, 31.7499504, -75.0265198, 31.5891190, -106.6621857, 106.7764587
35: -70.8468628, 35.0228195, -70.8010712, 34.8746490, -105.7214966, 105.8238831
36: -71.7779236, 37.6582947, -71.7420044, 37.4714203, -109.2493286, 109.4002991
37: -102.0846710, 33.2385712, -102.0042038, 33.1910934, -135.2757568, 135.2427673
38: -86.8634491, 50.4659576, -86.8189621, 50.2269211, -137.0903625, 137.2849121
39: -97.7229156, 43.8827782, -97.6407776, 43.8230972, -141.5460205, 141.5235596
40: -78.4205627, 34.3000107, -78.3276520, 34.2611237, -112.6816864, 112.6276627
41: -64.5665436, 40.6114731, -64.5014801, 40.5197296, -105.0862732, 105.1129532
42: -48.4914932, 35.9768143, -48.4553261, 35.9067307, -84.3982239, 84.4321442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9851546
time: 79.04 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0168436
time: 100.14 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -68.9159698, 38.8710670, -68.3795090, 38.7804375, -107.6964111, 107.2505798
1: -36.4381065, 36.6212463, -36.0739098, 36.5530396, -72.9911270, 72.6951447
2: -32.1789627, 38.6822128, -31.8787079, 38.6063194, -70.7852783, 70.5609131
3: -35.6943359, 43.0615425, -35.4278145, 42.9748192, -78.6691589, 78.4893570
4: -41.6818962, 40.3232994, -41.3068008, 40.2530212, -81.9349136, 81.6300964
5: -37.1750832, 42.5074577, -36.9029312, 42.3990211, -79.5741043, 79.4103851
6: -62.4264755, 37.5952568, -62.3337021, 37.3689346, -99.7954025, 99.9289551
7: -44.4422569, 40.5397339, -44.0547028, 40.4426575, -84.8849182, 84.5944366
8: -50.0814209, 46.6693115, -49.6495857, 46.5871582, -96.6685791, 96.3188934
9: -40.9563217, 43.9559784, -40.6188507, 43.8549728, -84.8112946, 84.5748215
10: -63.2964058, 58.5220680, -62.9391518, 58.3981705, -121.6945648, 121.4612122
11: -59.4414101, 33.6954536, -59.2697563, 33.5374069, -92.9788208, 92.9652100
12: -60.8070221, 42.4800797, -60.6940079, 42.2474365, -103.0544434, 103.1740875
13: -65.4664307, 61.3985405, -65.1465149, 61.2285156, -126.6949387, 126.5450516
14: -99.4963379, 46.7428703, -99.1035614, 46.6351624, -146.1315002, 145.8464203
15: -47.9007111, 43.1296616, -47.6722794, 42.9817657, -90.8824768, 90.8019409
16: -62.6970596, 45.8785667, -62.3159714, 45.7459641, -108.4430237, 108.1945343
17: -96.4485626, 43.8749619, -96.0426941, 43.7709198, -140.2194824, 139.9176636
18: -59.4782562, 48.0129547, -59.3475647, 47.7647629, -107.2430191, 107.3605118
19: -48.6604042, 28.0069313, -48.5351601, 27.7561760, -76.4165802, 76.5420837
20: -46.5934677, 32.1802025, -46.4967461, 31.9451809, -78.5386505, 78.6769485
21: -58.3573532, 32.8066254, -58.2083397, 32.5714188, -90.9287720, 91.0149536
22: -61.0869522, 34.7099724, -60.9197540, 34.4375610, -95.5244980, 95.6297302
23: -47.2084160, 35.6606064, -47.0877838, 35.4185677, -82.6269836, 82.7483902
24: -57.4516335, 34.0641937, -57.3202095, 33.8496666, -91.3012924, 91.3843918
25: -51.6163139, 37.4566650, -51.4949226, 37.1735191, -88.7898331, 88.9515839
26: -70.0963745, 50.3344040, -69.9185638, 49.9957619, -120.0921326, 120.2529678
27: -57.0349121, 38.9467468, -56.8811302, 38.6810303, -95.7159424, 95.8278809
28: -47.6909447, 39.3589859, -47.5681305, 39.0385513, -86.7294922, 86.9271164
29: -60.2665062, 30.6342564, -60.0926819, 30.3835487, -90.6500549, 90.7269363
30: -58.4751434, 40.5900574, -58.3645897, 40.4285622, -98.9037018, 98.9546432
31: -59.5586166, 34.4773941, -59.4119682, 34.1534500, -93.7120514, 93.8893585
32: -61.0985336, 35.8803864, -60.9968185, 35.6863594, -96.7848969, 96.8771896
33: -86.5119934, 46.4412956, -86.3427811, 46.1930389, -132.7050323, 132.7840576
34: -75.1984253, 32.0424271, -75.0545349, 31.7296028, -106.9280167, 107.0969620
35: -70.9819031, 35.3086433, -70.8332596, 35.0121689, -105.9940643, 106.1419067
36: -71.9089966, 37.9926987, -71.7678223, 37.6323090, -109.5413055, 109.7605209
37: -102.2503967, 33.4794083, -102.0513916, 33.3066864, -135.5570679, 135.5307922
38: -87.0461655, 50.9022522, -86.8551025, 50.4358902, -137.4820404, 137.7573547
39: -97.8792572, 44.0753517, -97.6925735, 43.9151344, -141.7943878, 141.7679291
40: -78.5401154, 34.5034790, -78.3740768, 34.3578682, -112.8979797, 112.8775558
41: -64.6693192, 40.8314438, -64.5323486, 40.6236877, -105.2930069, 105.3637848
42: -48.5610580, 36.1734924, -48.4780121, 35.9996147, -84.5606689, 84.6515045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9901267
time: 102.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0207735
time: 102.22 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -68.4992142, 38.7996368, -68.2522430, 38.7383308, -107.2375488, 107.0518723
1: -36.1764526, 36.5636139, -36.0064850, 36.5307007, -72.7071533, 72.5700989
2: -31.9448624, 38.6166611, -31.7923813, 38.5858154, -70.5306778, 70.4090424
3: -35.4741898, 42.9833527, -35.3449707, 42.9262047, -78.4003906, 78.3283234
4: -41.4082031, 40.2643433, -41.2390823, 40.2304077, -81.6386108, 81.5034256
5: -36.9457016, 42.4212875, -36.8357544, 42.3744774, -79.3201675, 79.2570419
6: -62.3538513, 37.2927017, -62.2814789, 37.1373291, -99.4911652, 99.5741806
7: -44.1682320, 40.4507217, -44.0191841, 40.4452438, -84.6134796, 84.4699020
8: -49.7363205, 46.5956802, -49.5223274, 46.5443344, -96.2806549, 96.1180038
9: -40.6924438, 43.8652687, -40.5485840, 43.8442917, -84.5367355, 84.4138489
10: -62.9940948, 58.4080200, -62.8227005, 58.3542480, -121.3483276, 121.2307205
11: -59.3312645, 33.5520020, -59.2074165, 33.4329681, -92.7642365, 92.7594147
12: -60.7181969, 42.2267380, -60.6466370, 42.1705780, -102.8887787, 102.8733749
13: -65.2182617, 61.2975311, -65.0487671, 61.1908836, -126.4091492, 126.3462753
14: -99.1184311, 46.6460037, -98.8909912, 46.5750351, -145.6934662, 145.5369873
15: -47.6556206, 43.0288277, -47.4916267, 42.9648285, -90.6204376, 90.5204468
16: -62.4314384, 45.7621078, -62.2677002, 45.7451057, -108.1765442, 108.0298080
17: -96.1721420, 43.7865028, -95.9662781, 43.6723747, -139.8445129, 139.7527771
18: -59.3738136, 47.7733040, -59.3053513, 47.6333084, -107.0071182, 107.0786514
19: -48.5679855, 27.8061123, -48.4655075, 27.6398563, -76.2078400, 76.2716217
20: -46.5106621, 31.9610405, -46.4091873, 31.7922344, -78.3028946, 78.3702240
21: -58.2510986, 32.5764389, -58.1132965, 32.3840714, -90.6351700, 90.6897354
22: -60.9634361, 34.4972610, -60.8671989, 34.3379593, -95.3013916, 95.3644562
23: -47.1292114, 35.4922638, -47.0412636, 35.3669853, -82.4961929, 82.5335236
24: -57.3701286, 33.9049988, -57.2513275, 33.7737808, -91.1439056, 91.1563263
25: -51.5102196, 37.2429733, -51.4399147, 37.0648117, -88.5750198, 88.6828842
26: -69.9645691, 50.0309830, -69.8851624, 49.8574181, -119.8219833, 119.9161453
27: -56.9406433, 38.7355042, -56.8155403, 38.5638351, -95.5044556, 95.5510406
28: -47.6010933, 39.1255836, -47.5284424, 38.9502869, -86.5513763, 86.6540222
29: -60.1570930, 30.4691696, -60.0426178, 30.3359375, -90.4930267, 90.5117722
30: -58.3679733, 40.4167480, -58.2547722, 40.2569199, -98.6248932, 98.6715012
31: -59.4486008, 34.2236023, -59.3521233, 34.0348244, -93.4834290, 93.5757217
32: -61.0100899, 35.6498642, -60.9417267, 35.5349884, -96.5450745, 96.5915833
33: -86.3685150, 46.1626663, -86.2954712, 46.0566025, -132.4251099, 132.4581299
34: -75.0708923, 31.7715359, -75.0478439, 31.6386642, -106.7095566, 106.8193817
35: -70.8538055, 35.0502396, -70.8321609, 34.9328728, -105.7866821, 105.8823929
36: -71.7814636, 37.6784286, -71.7479248, 37.5167618, -109.2982254, 109.4263535
37: -102.0834885, 33.2635880, -101.9815521, 33.2343407, -135.3178253, 135.2451324
38: -86.8642273, 50.5044632, -86.8348160, 50.3089447, -137.1731720, 137.3392792
39: -97.7260895, 43.8765526, -97.6173782, 43.8043518, -141.5304413, 141.4939270
40: -78.4291229, 34.3223877, -78.3282242, 34.3021545, -112.7312698, 112.6506119
41: -64.5714417, 40.6060181, -64.5081482, 40.5080643, -105.0794983, 105.1141586
42: -48.4863472, 35.9868736, -48.4522743, 35.9200745, -84.4064102, 84.4391479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9767693
time: 159.21 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0153350
time: 83.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -68.7001114, 38.8261833, -68.6745071, 38.8537483, -107.5538635, 107.5006866
1: -36.2950058, 36.5777435, -36.2520752, 36.6116333, -72.9066315, 72.8298187
2: -32.0594864, 38.6322479, -32.0277710, 38.6683731, -70.7278595, 70.6600189
3: -35.5795593, 43.0125847, -35.5623016, 43.0406914, -78.6202469, 78.5748901
4: -41.5416489, 40.2860870, -41.5158958, 40.3078690, -81.8495178, 81.8019791
5: -37.0504189, 42.4494400, -37.0520439, 42.4878883, -79.5382996, 79.5014801
6: -62.3864822, 37.4635124, -62.3996353, 37.4927521, -99.8792343, 99.8631439
7: -44.2734909, 40.4675026, -44.2415695, 40.5265923, -84.8000793, 84.7090683
8: -49.8916473, 46.6173782, -49.8437233, 46.6428909, -96.5345154, 96.4610977
9: -40.8158340, 43.8890038, -40.8050232, 43.9375610, -84.7533951, 84.6940231
10: -63.1366081, 58.4520378, -63.1188126, 58.4910278, -121.6276398, 121.5708466
11: -59.3693581, 33.6554489, -59.3561821, 33.6486359, -93.0179749, 93.0116272
12: -60.7714577, 42.3033066, -60.7666321, 42.3553925, -103.1268463, 103.0699234
13: -65.3701553, 61.3448143, -65.3706055, 61.3563080, -126.7264633, 126.7154160
14: -99.3370056, 46.6799316, -99.3581848, 46.7194481, -146.0564575, 146.0381165
15: -47.7996674, 43.0722809, -47.7924309, 43.0817947, -90.8814545, 90.8647079
16: -62.5466270, 45.7929802, -62.5191765, 45.8574791, -108.4041061, 108.3121490
17: -96.3178482, 43.8215790, -96.2874603, 43.8379211, -140.1557617, 140.1090393
18: -59.4122391, 47.8895760, -59.4298058, 47.8749237, -107.2871628, 107.3193817
19: -48.6044312, 27.9320850, -48.6262970, 27.8955269, -76.4999542, 76.5583801
20: -46.5435753, 32.0979996, -46.5729179, 32.0717621, -78.6153259, 78.6709137
21: -58.2982330, 32.7340317, -58.3067780, 32.7037201, -91.0019531, 91.0408096
22: -60.9975471, 34.6098557, -61.0457802, 34.5696220, -95.5671692, 95.6556320
23: -47.1624298, 35.5967560, -47.1801643, 35.5801926, -82.7426224, 82.7769165
24: -57.4023857, 34.0056381, -57.4236717, 33.9778137, -91.3802032, 91.4292908
25: -51.5444756, 37.3623047, -51.5874214, 37.3116989, -88.8561707, 88.9497223
26: -70.0032196, 50.1921997, -70.0624695, 50.1827431, -120.1859589, 120.2546692
27: -56.9785385, 38.8718567, -56.9969444, 38.8407478, -95.8192825, 95.8688049
28: -47.6301651, 39.2591743, -47.6716995, 39.2194557, -86.8496094, 86.9308701
29: -60.1905632, 30.5603733, -60.2151680, 30.5252533, -90.7158051, 90.7755356
30: -58.4078789, 40.5467949, -58.4306068, 40.5254745, -98.9333344, 98.9773941
31: -59.4899597, 34.3596687, -59.5124207, 34.3124847, -93.8024445, 93.8720779
32: -61.0515785, 35.7616348, -61.0681572, 35.7798424, -96.8314056, 96.8297882
33: -86.4247894, 46.2854919, -86.4668808, 46.3150177, -132.7398071, 132.7523651
34: -75.1069183, 31.8900661, -75.1669464, 31.8821621, -106.9890823, 107.0570145
35: -70.8873291, 35.1480522, -70.9509583, 35.1375465, -106.0248718, 106.0990143
36: -71.8094788, 37.8061295, -71.8821869, 37.7802429, -109.5897217, 109.6883163
37: -102.1436996, 33.3461113, -102.1957474, 33.4088287, -135.5525208, 135.5418396
38: -86.9042587, 50.6541557, -87.0056000, 50.6250381, -137.5292969, 137.6597595
39: -97.7939148, 43.9436493, -97.8263474, 43.9556313, -141.7495422, 141.7699890
40: -78.4803314, 34.3747635, -78.4711914, 34.4185638, -112.8988953, 112.8459549
41: -64.6085815, 40.7329636, -64.6329498, 40.7718811, -105.3804474, 105.3659134
42: -48.5217934, 36.0850868, -48.5353622, 36.1306000, -84.6523895, 84.6204376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0147187
time: 86.47 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0532279
time: 99.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -68.9582977, 38.8730469, -68.4693375, 38.7605858, -107.7188797, 107.3423767
1: -36.4747849, 36.6200180, -36.1492004, 36.5405655, -73.0153503, 72.7692184
2: -32.1959763, 38.6808319, -31.9131126, 38.5975418, -70.7935181, 70.5939484
3: -35.7090836, 43.0545197, -35.4580231, 42.9457703, -78.6548462, 78.5125427
4: -41.7216530, 40.3194122, -41.3890190, 40.2456779, -81.9673309, 81.7084274
5: -37.2016830, 42.5022545, -36.9586906, 42.3927994, -79.5944748, 79.4609375
6: -62.4328804, 37.5362740, -62.3102303, 37.2521324, -99.6850128, 99.8465042
7: -44.5086098, 40.5392647, -44.1835289, 40.4603386, -84.9689407, 84.7227936
8: -50.1050148, 46.6720810, -49.6993332, 46.5655251, -96.6705322, 96.3714142
9: -40.9865837, 43.9527588, -40.6885452, 43.8626976, -84.8492737, 84.6412964
10: -63.3101997, 58.5166359, -62.9727020, 58.3876495, -121.6978455, 121.4893341
11: -59.4712906, 33.6724396, -59.2730064, 33.4892769, -92.9605713, 92.9454498
12: -60.8012581, 42.5122185, -60.6797142, 42.3027878, -103.1040497, 103.1919250
13: -65.4818573, 61.3995171, -65.1735229, 61.2311859, -126.7130280, 126.5730362
14: -99.4770508, 46.7346764, -99.0591431, 46.5971069, -146.0741577, 145.7938232
15: -47.8533936, 43.1640282, -47.5820045, 43.0280914, -90.8814850, 90.7460327
16: -62.7481651, 45.8728485, -62.4175453, 45.7681465, -108.5163116, 108.2903748
17: -96.4869080, 43.8754959, -96.1132126, 43.7053757, -140.1922913, 139.9887085
18: -59.4842072, 48.0012894, -59.3441887, 47.7428932, -107.2270966, 107.3454666
19: -48.6564980, 27.9959240, -48.4929848, 27.7328663, -76.3893661, 76.4889069
20: -46.5892677, 32.1490326, -46.4325256, 31.8825855, -78.4718475, 78.5815582
21: -58.3606987, 32.7564850, -58.1561127, 32.4710121, -90.8317032, 90.9125977
22: -61.0946999, 34.7147446, -60.9037056, 34.4435959, -95.5382996, 95.6184540
23: -47.2043190, 35.6776886, -47.0657234, 35.4561310, -82.6604462, 82.7434082
24: -57.4609528, 34.0627975, -57.2834702, 33.8500671, -91.3110199, 91.3462677
25: -51.6135941, 37.4548874, -51.4677582, 37.1670952, -88.7806854, 88.9226456
26: -70.1061020, 50.3351135, -69.9218292, 50.0046692, -120.1107635, 120.2569427
27: -57.0409508, 38.9340096, -56.8516579, 38.6589050, -95.6998596, 95.7856598
28: -47.6865768, 39.3712997, -47.5486259, 39.0682449, -86.7548218, 86.9199219
29: -60.2817688, 30.6559772, -60.0846977, 30.4260235, -90.7077866, 90.7406769
30: -58.4847221, 40.5297432, -58.3064117, 40.3086166, -98.7933197, 98.8361511
31: -59.5606079, 34.4795914, -59.3903770, 34.1588593, -93.7194519, 93.8699646
32: -61.0988617, 35.8625832, -60.9738121, 35.6348686, -96.7337341, 96.8363953
33: -86.5137482, 46.4337463, -86.3407593, 46.1853027, -132.6990509, 132.7744751
34: -75.1961212, 32.0639839, -75.0759964, 31.7792244, -106.9753418, 107.1399765
35: -70.9885559, 35.3364220, -70.8643723, 35.0708313, -106.0593719, 106.2007904
36: -71.9123688, 38.0131111, -71.7736816, 37.6781578, -109.5905304, 109.7867889
37: -102.2489853, 33.5047073, -102.0289078, 33.3505630, -135.5995483, 135.5336151
38: -87.0468140, 50.9408684, -86.8710632, 50.5183334, -137.5651550, 137.8119202
39: -97.8818512, 44.0690193, -97.6691971, 43.8963852, -141.7782288, 141.7382202
40: -78.5489044, 34.5263519, -78.3755035, 34.3981247, -112.9470291, 112.9018555
41: -64.6740646, 40.8264465, -64.5391235, 40.6128006, -105.2868652, 105.3655624
42: -48.5560036, 36.1837196, -48.4751129, 36.0134888, -84.5694885, 84.6588287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9817220
time: 93.12 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0192334
time: 92.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -69.1592712, 38.8995247, -68.8916702, 38.8759384, -108.0352097, 107.7911911
1: -36.5933533, 36.6341476, -36.3948174, 36.6214600, -73.2148132, 73.0289612
2: -32.3105850, 38.6963882, -32.1485100, 38.6800652, -70.9906464, 70.8448944
3: -35.8144531, 43.0837402, -35.6753426, 43.0602417, -78.8746948, 78.7590790
4: -41.8552170, 40.3411179, -41.6659164, 40.3231125, -82.1783295, 82.0070343
5: -37.3064194, 42.5303917, -37.1749802, 42.5061493, -79.8125610, 79.7053680
6: -62.4655113, 37.7071953, -62.4283447, 37.6076393, -100.0731506, 100.1355438
7: -44.6138992, 40.5560150, -44.4059258, 40.5416412, -85.1555405, 84.9619370
8: -50.2604065, 46.6937943, -50.0207901, 46.6641388, -96.9245300, 96.7145691
9: -41.1100998, 43.9764671, -40.9450874, 43.9559479, -85.0660477, 84.9215546
10: -63.4527817, 58.5606194, -63.2689133, 58.5243912, -121.9771652, 121.8295288
11: -59.5094337, 33.7760010, -59.4217834, 33.7050247, -93.2144547, 93.1977844
12: -60.8545151, 42.5890503, -60.7997894, 42.4877319, -103.3422394, 103.3888397
13: -65.6337585, 61.4467659, -65.4953995, 61.3964996, -127.0302582, 126.9421616
14: -99.6956558, 46.7685547, -99.5264053, 46.7414703, -146.4371338, 146.2949524
15: -47.9976196, 43.2075462, -47.8830566, 43.1450806, -91.1426773, 91.0905991
16: -62.8633614, 45.9036407, -62.6690445, 45.8804321, -108.7437897, 108.5726852
17: -96.6326675, 43.9105988, -96.4345093, 43.8709145, -140.5035858, 140.3451080
18: -59.5225906, 48.1175690, -59.4686089, 47.9845123, -107.5070953, 107.5861816
19: -48.6928978, 28.1219406, -48.6537323, 27.9885139, -76.6814041, 76.7756653
20: -46.6221771, 32.2860031, -46.5962334, 32.1621399, -78.7843170, 78.8822327
21: -58.4078484, 32.9141121, -58.3495789, 32.7907104, -91.1985474, 91.2636719
22: -61.1287537, 34.8273735, -61.0822296, 34.6752548, -95.8040085, 95.9096069
23: -47.2374954, 35.7822342, -47.2045631, 35.6694107, -82.9069061, 82.9867859
24: -57.4932175, 34.1634674, -57.4558144, 34.0541153, -91.5473328, 91.6192780
25: -51.6477776, 37.5742340, -51.6151810, 37.4140396, -89.0618134, 89.1894150
26: -70.1446915, 50.4964104, -70.0990448, 50.3300552, -120.4747467, 120.5954590
27: -57.0788803, 39.0704384, -57.0330124, 38.9359055, -96.0147858, 96.1034546
28: -47.7155838, 39.5049095, -47.6918221, 39.3374176, -87.0530014, 87.1967316
29: -60.3151932, 30.7471809, -60.2572403, 30.6153679, -90.9305573, 91.0044174
30: -58.5246391, 40.6598358, -58.4822388, 40.5772324, -99.1018677, 99.1420670
31: -59.6019020, 34.6156883, -59.5506096, 34.4365540, -94.0384521, 94.1662979
32: -61.1402473, 35.9743767, -61.1001320, 35.8797989, -97.0200424, 97.0744934
33: -86.5699768, 46.5566635, -86.5119934, 46.4437675, -133.0137329, 133.0686493
34: -75.2320633, 32.1824875, -75.1950302, 32.0227432, -107.2548065, 107.3775177
35: -71.0220032, 35.4342575, -70.9831161, 35.2754784, -106.2974701, 106.4173660
36: -71.9402618, 38.1408882, -71.9078903, 37.9416885, -109.8819427, 110.0487747
37: -102.3091660, 33.5873146, -102.2430573, 33.5250931, -135.8342590, 135.8303680
38: -87.0867615, 51.0905914, -87.0417862, 50.8344650, -137.9212036, 138.1323700
39: -97.9494629, 44.1361084, -97.8778992, 44.0476456, -141.9971008, 142.0140076
40: -78.6000366, 34.5787277, -78.5184784, 34.5145111, -113.1145477, 113.0971985
41: -64.7111435, 40.9534378, -64.6638031, 40.8766899, -105.5878296, 105.6172333
42: -48.5914268, 36.2819939, -48.5582047, 36.2240753, -84.8154907, 84.8401947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=435, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0571066, upper bound: 51.0196642
time: 102.25 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0571064
time: 134.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 239.06 seconds
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9771141
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0042155
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9558195
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9817063
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9558195
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9817063
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9851546
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0168436
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9901267
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0207735
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9767693
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0153350
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0147187
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0532279
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9817220
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0192334
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0571066, upper bound: 51.0196642
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 239.06
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0571064

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -68.7385178, 38.8180847, -68.1563034, 38.7553940, -107.4939117, 106.9743729
1: -36.3476715, 36.5717239, -35.9281502, 36.5402527, -72.8879242, 72.4998779
2: -32.1006432, 38.6273346, -31.7548180, 38.5915337, -70.6921768, 70.3821564
3: -35.6264572, 43.0089912, -35.3122711, 42.9523430, -78.5787964, 78.3212585
4: -41.5832672, 40.2805481, -41.1525269, 40.2344513, -81.8177185, 81.4330750
5: -37.0951309, 42.4419785, -36.7767181, 42.3769913, -79.4721222, 79.2186966
6: -62.3695602, 37.5096970, -62.2990837, 37.2489014, -99.6184616, 99.8087769
7: -44.3368492, 40.4604683, -43.8866043, 40.4230042, -84.7598572, 84.3470688
8: -49.9882698, 46.6134644, -49.4693298, 46.5621338, -96.5503998, 96.0827942
9: -40.8500748, 43.8868065, -40.4760551, 43.8326950, -84.6827698, 84.3628616
10: -63.1915703, 58.4501648, -62.7858696, 58.3600464, -121.5516129, 121.2360306
11: -59.3839493, 33.6555786, -59.2017212, 33.4780502, -92.8619995, 92.8572998
12: -60.7552109, 42.3875237, -60.6570969, 42.1120834, -102.8672943, 103.0446091
13: -65.3323059, 61.3430786, -65.0163193, 61.1844864, -126.5167923, 126.3593979
14: -99.3578568, 46.6725655, -98.9256134, 46.6051636, -145.9630127, 145.5981750
15: -47.8624382, 43.0370903, -47.5774956, 42.9166489, -90.7790833, 90.6145782
16: -62.5779343, 45.7917595, -62.1614761, 45.7186279, -108.2965622, 107.9532318
17: -96.3336945, 43.8201447, -95.8857727, 43.7350693, -140.0687561, 139.7059174
18: -59.4137115, 47.9327850, -59.3043098, 47.6533966, -107.0671082, 107.2370834
19: -48.6044159, 27.9509659, -48.5033188, 27.6611824, -76.2655945, 76.4542847
20: -46.5383072, 32.1216965, -46.4696465, 31.8518105, -78.3901062, 78.5913391
21: -58.3022690, 32.7550507, -58.1601906, 32.4820518, -90.7843170, 90.9152374
22: -60.9962196, 34.6267548, -60.8773766, 34.3305664, -95.3267822, 95.5041351
23: -47.1581612, 35.5897446, -47.0593987, 35.3266411, -82.4848022, 82.6491394
24: -57.3835945, 33.9940643, -57.2829895, 33.7722549, -91.1558533, 91.2770538
25: -51.5414810, 37.3892708, -51.4624252, 37.0691948, -88.6106720, 88.8516998
26: -69.9896164, 50.2131195, -69.8762512, 49.8451614, -119.8347778, 120.0893707
27: -56.9714851, 38.8467827, -56.8405724, 38.5845947, -95.5560760, 95.6873474
28: -47.6230698, 39.2721100, -47.5437889, 38.9180984, -86.5411682, 86.8158875
29: -60.1902161, 30.5536537, -60.0447693, 30.2918701, -90.4820709, 90.5984192
30: -58.4119911, 40.5429878, -58.3088226, 40.3750916, -98.7870636, 98.8518066
31: -59.4880943, 34.3974609, -59.3690910, 34.0269203, -93.5149994, 93.7665558
32: -61.0374069, 35.8071365, -60.9597130, 35.5829849, -96.6203918, 96.7668457
33: -86.4126205, 46.3619766, -86.2918396, 46.0611496, -132.4737701, 132.6538086
34: -75.0949326, 31.9319267, -75.0205688, 31.5855751, -106.6805115, 106.9524918
35: -70.8716278, 35.2177620, -70.7947540, 34.8720207, -105.7436371, 106.0125122
36: -71.7978516, 37.8950195, -71.7357101, 37.4687653, -109.2666168, 109.6307297
37: -102.1347580, 33.3984070, -101.9965515, 33.1878662, -135.3226318, 135.3949585
38: -86.8956909, 50.7844353, -86.8108826, 50.2230911, -137.1187744, 137.5953217
39: -97.7738037, 44.0382004, -97.6336136, 43.8200912, -141.5939026, 141.6718140
40: -78.4713669, 34.4172974, -78.3223572, 34.2568245, -112.7281876, 112.7396393
41: -64.5980606, 40.7432442, -64.4969788, 40.5146523, -105.1127167, 105.2402191
42: -48.5146141, 36.0909882, -48.4504204, 35.9001389, -84.4147491, 84.5414124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=434, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1653

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1773

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9986665
time: 106.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0148196
time: 118.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -69.1958389, 38.8913574, -68.3715439, 38.7775192, -107.9733582, 107.2629013
1: -36.6448936, 36.6282730, -36.0701294, 36.5500488, -73.1949463, 72.6984024
2: -32.3510437, 38.6915588, -31.8747101, 38.6031303, -70.9541702, 70.5662689
3: -35.8610840, 43.0801430, -35.4244919, 42.9716873, -78.8327713, 78.5046387
4: -41.8952484, 40.3358269, -41.3011971, 40.2495193, -82.1447601, 81.6370239
5: -37.3510361, 42.5228539, -36.8988724, 42.3951645, -79.7462006, 79.4217224
6: -62.4465523, 37.7513390, -62.3266678, 37.3624268, -99.8089752, 100.0780029
7: -44.6772881, 40.5487938, -44.0502853, 40.4379959, -85.1152802, 84.5990753
8: -50.3558960, 46.6898575, -49.6449852, 46.5831795, -96.9390717, 96.3348389
9: -41.1432953, 43.9741516, -40.6151047, 43.8509903, -84.9942780, 84.5892563
10: -63.5067368, 58.5583954, -62.9347649, 58.3933640, -121.9001007, 121.4931641
11: -59.5290985, 33.7745972, -59.2655334, 33.5332794, -93.0623779, 93.0401306
12: -60.8354645, 42.6740074, -60.6889610, 42.2429771, -103.0784378, 103.3629608
13: -65.5928879, 61.4445992, -65.1393738, 61.2246246, -126.8175125, 126.5839691
14: -99.7146530, 46.7602234, -99.0927429, 46.6272850, -146.3419189, 145.8529663
15: -48.0598221, 43.1724930, -47.6668777, 42.9792633, -91.0390854, 90.8393631
16: -62.8949776, 45.9018173, -62.3107834, 45.7415352, -108.6364975, 108.2125931
17: -96.6468964, 43.9091873, -96.0340729, 43.7676773, -140.4145508, 139.9432526
18: -59.5249939, 48.1590614, -59.3431473, 47.7616653, -107.2866287, 107.5021973
19: -48.6931725, 28.1390991, -48.5307693, 27.7529640, -76.4461365, 76.6698685
20: -46.6173668, 32.3083267, -46.4927635, 31.9411221, -78.5584869, 78.8010864
21: -58.4121742, 32.9335060, -58.2029495, 32.5677795, -90.9799500, 91.1364517
22: -61.1262817, 34.8437958, -60.9138680, 34.4351997, -95.5614700, 95.7576599
23: -47.2354774, 35.7735901, -47.0836945, 35.4145508, -82.6500244, 82.8572845
24: -57.4752197, 34.1504478, -57.3145752, 33.8470955, -91.3223114, 91.4650192
25: -51.6449699, 37.6003456, -51.4900131, 37.1702843, -88.8152390, 89.0903625
26: -70.1314545, 50.5159683, -69.9127045, 49.9911804, -120.1226349, 120.4286728
27: -57.0711555, 39.0434608, -56.8765907, 38.6781578, -95.7493134, 95.9200439
28: -47.7092285, 39.5161514, -47.5637856, 39.0346909, -86.7439194, 87.0799332
29: -60.3138504, 30.7395000, -60.0868530, 30.3809299, -90.6947784, 90.8263397
30: -58.5287514, 40.6541367, -58.3601723, 40.4250412, -98.9537964, 99.0143051
31: -59.6012421, 34.6516304, -59.4069023, 34.1495018, -93.7507324, 94.0585327
32: -61.1248550, 36.0198288, -60.9913063, 35.6819267, -96.8067780, 97.0111389
33: -86.5578613, 46.6330299, -86.3369293, 46.1892586, -132.7471161, 132.9699554
34: -75.2202606, 32.2238960, -75.0485153, 31.7258530, -106.9461136, 107.2724152
35: -71.0063477, 35.5034409, -70.8268280, 35.0094757, -106.0158157, 106.3302689
36: -71.9282990, 38.2294807, -71.7614441, 37.6296082, -109.5579071, 109.9909210
37: -102.3004684, 33.6405144, -102.0436554, 33.3032761, -135.6037445, 135.6841736
38: -87.0780869, 51.2212219, -86.8469315, 50.4319496, -137.5100098, 138.0681458
39: -97.9288788, 44.2313156, -97.6852570, 43.9120903, -141.8409729, 141.9165649
40: -78.5903397, 34.6177368, -78.3686981, 34.3531837, -112.9435272, 112.9864349
41: -64.7003326, 40.9624062, -64.5277328, 40.6181946, -105.3185272, 105.4901276
42: -48.5834045, 36.2858429, -48.4730721, 35.9925232, -84.5759201, 84.7589111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=434, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1773

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0027793
time: 90.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9808316
time: 100.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -68.7803955, 38.8200760, -68.2447433, 38.7355194, -107.5159073, 107.0648193
1: -36.3842392, 36.5705032, -36.0029526, 36.5277748, -72.9120026, 72.5734558
2: -32.1173515, 38.6259575, -31.7886124, 38.5826645, -70.7000122, 70.4145660
3: -35.6407585, 43.0020218, -35.3418198, 42.9231071, -78.5638657, 78.3438339
4: -41.6224442, 40.2765999, -41.2338409, 40.2269783, -81.8494263, 81.5104370
5: -37.1213799, 42.4367828, -36.8319168, 42.3706818, -79.4920502, 79.2686996
6: -62.3753777, 37.4501228, -62.2741547, 37.1313286, -99.5066986, 99.7242737
7: -44.4030151, 40.4599838, -44.0150452, 40.4406548, -84.8436737, 84.4750290
8: -50.0110970, 46.6163483, -49.5179482, 46.5404396, -96.5515366, 96.1342926
9: -40.8798866, 43.8836136, -40.5450706, 43.8404083, -84.7202911, 84.4286804
10: -63.2045975, 58.4447746, -62.8184357, 58.3496094, -121.5541916, 121.2632141
11: -59.4136734, 33.6319962, -59.2032700, 33.4291229, -92.8427963, 92.8352661
12: -60.7484283, 42.4195938, -60.6408768, 42.1664124, -102.9148407, 103.0604553
13: -65.3467255, 61.3438606, -65.0420380, 61.1871567, -126.5338593, 126.3858871
14: -99.3379745, 46.6642799, -98.8808517, 46.5683250, -145.9062958, 145.5451355
15: -47.8148575, 43.0712433, -47.4865150, 42.9624367, -90.7772980, 90.5577545
16: -62.6290741, 45.7860603, -62.2627258, 45.7407684, -108.3698273, 108.0487823
17: -96.3730698, 43.8204384, -95.9581299, 43.6692619, -140.0423279, 139.7785645
18: -59.4198761, 47.9201126, -59.3010941, 47.6303024, -107.0501785, 107.2211990
19: -48.6005402, 27.9391708, -48.4611969, 27.6368523, -76.2373810, 76.4003601
20: -46.5340691, 32.0901413, -46.4052963, 31.7884007, -78.3224640, 78.4954376
21: -58.3056602, 32.7040634, -58.1080704, 32.3805847, -90.6862488, 90.8121262
22: -61.0040398, 34.6307831, -60.8615074, 34.3356209, -95.3396606, 95.4922943
23: -47.1541672, 35.6059952, -47.0372696, 35.3632088, -82.5173798, 82.6432648
24: -57.3931808, 33.9915504, -57.2459183, 33.7712479, -91.1644287, 91.2374725
25: -51.5387840, 37.3866730, -51.4351044, 37.0616531, -88.6004333, 88.8217773
26: -69.9994659, 50.2132568, -69.8794250, 49.8530807, -119.8525467, 120.0926819
27: -56.9774666, 38.8328400, -56.8112450, 38.5610504, -95.5385056, 95.6440887
28: -47.6186829, 39.2835579, -47.5241928, 38.9466171, -86.5653000, 86.8077545
29: -60.2056351, 30.5746078, -60.0369720, 30.3334656, -90.5390930, 90.6115723
30: -58.4215279, 40.4812012, -58.2505646, 40.2535019, -98.6750183, 98.7317657
31: -59.4900703, 34.3987122, -59.3471832, 34.0310745, -93.5211487, 93.7458954
32: -61.0374756, 35.7892036, -60.9363060, 35.5308609, -96.5683365, 96.7254944
33: -86.4145813, 46.3542480, -86.2899017, 46.0529213, -132.4674988, 132.6441498
34: -75.0927582, 31.9535294, -75.0419922, 31.6351414, -106.7278900, 106.9955139
35: -70.8785095, 35.2453079, -70.8259048, 34.9302750, -105.8087845, 106.0712051
36: -71.8013611, 37.9152603, -71.7416534, 37.5141449, -109.3154907, 109.6569061
37: -102.1333847, 33.4236450, -101.9739609, 33.2310638, -135.3644409, 135.3976135
38: -86.8964233, 50.8230896, -86.8267899, 50.3051796, -137.2015839, 137.6498718
39: -97.7769623, 44.0320625, -97.6103363, 43.8014259, -141.5783844, 141.6423950
40: -78.4797821, 34.4403458, -78.3229904, 34.2977982, -112.7775803, 112.7633362
41: -64.6028900, 40.7376404, -64.5037003, 40.5029602, -105.1058502, 105.2413406
42: -48.5092964, 36.1012573, -48.4477730, 35.9134979, -84.4227905, 84.5490265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=434, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1653

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1773

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9843979
time: 86.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0146498
time: 82.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -68.6683426, 38.7928581, -68.6616211, 38.8400269, -107.5083694, 107.4544830
1: -36.2804642, 36.5299835, -36.2462120, 36.5919304, -72.8723907, 72.7761993
2: -32.0475998, 38.5791397, -32.0229683, 38.6465149, -70.6941147, 70.6021118
3: -35.5685692, 42.9688606, -35.5578194, 43.0225525, -78.5911102, 78.5266724
4: -41.5238419, 40.2446442, -41.5087700, 40.2902336, -81.8140717, 81.7534180
5: -37.0376854, 42.3958054, -37.0469131, 42.4657860, -79.5034714, 79.4427185
6: -62.3475113, 37.4415512, -62.3840828, 37.4839592, -99.8314667, 99.8256226
7: -44.2590599, 40.3969040, -44.2357635, 40.4975471, -84.7565918, 84.6326675
8: -49.8740349, 46.5583038, -49.8365746, 46.6185303, -96.4925613, 96.3948669
9: -40.7992096, 43.8292999, -40.7982750, 43.9130554, -84.7122498, 84.6275787
10: -63.1153450, 58.3974724, -63.1100388, 58.4682770, -121.5836182, 121.5075073
11: -59.3444977, 33.6410751, -59.3459816, 33.6428375, -92.9873199, 92.9870529
12: -60.7415466, 42.2786369, -60.7541122, 42.3451920, -103.0867386, 103.0327454
13: -65.3449097, 61.3133659, -65.3604279, 61.3434372, -126.6883392, 126.6737976
14: -99.3004074, 46.6135902, -99.3432999, 46.6909943, -145.9913940, 145.9568787
15: -47.7738533, 43.0581627, -47.7819290, 43.0758781, -90.8497162, 90.8400879
16: -62.5229568, 45.7245255, -62.5091896, 45.8294373, -108.3523865, 108.2337112
17: -96.2835846, 43.7934952, -96.2734680, 43.8263474, -140.1099243, 140.0669556
18: -59.3702545, 47.8764191, -59.4118958, 47.8695984, -107.2398453, 107.2883148
19: -48.5542412, 27.9257298, -48.6053429, 27.8930492, -76.4472885, 76.5310669
20: -46.4972801, 32.0848846, -46.5537987, 32.0664215, -78.5636902, 78.6386871
21: -58.2540359, 32.7226105, -58.2884789, 32.6991272, -90.9531631, 91.0110779
22: -60.9247284, 34.6012115, -61.0158348, 34.5660553, -95.4907837, 95.6170502
23: -47.1218605, 35.5841408, -47.1626968, 35.5751495, -82.6970062, 82.7468414
24: -57.3591118, 33.9973106, -57.4053154, 33.9744110, -91.3335114, 91.4026260
25: -51.4797745, 37.3504639, -51.5606499, 37.3068733, -88.7866440, 88.9111176
26: -69.9291000, 50.1785431, -70.0317917, 50.1772461, -120.1063461, 120.2103348
27: -56.9406967, 38.8605728, -56.9813309, 38.8362160, -95.7769165, 95.8419037
28: -47.5749359, 39.2463913, -47.6489067, 39.2143211, -86.7892609, 86.8952942
29: -60.1352310, 30.5519104, -60.1919212, 30.5218716, -90.6571045, 90.7438354
30: -58.3846092, 40.5284653, -58.4209404, 40.5180016, -98.9026108, 98.9493942
31: -59.4325714, 34.3490334, -59.4883385, 34.3082886, -93.7408600, 93.8373566
32: -61.0108986, 35.7438965, -61.0504494, 35.7724762, -96.7833710, 96.7943268
33: -86.3533783, 46.2688065, -86.4370422, 46.3081818, -132.6615601, 132.7058411
34: -75.0299377, 31.8765030, -75.1352310, 31.8766747, -106.9066010, 107.0117340
35: -70.7993774, 35.1376953, -70.9147034, 35.1333466, -105.9327240, 106.0523987
36: -71.7092209, 37.7950134, -71.8410263, 37.7757111, -109.4849319, 109.6360397
37: -102.0508652, 33.3347549, -102.1571655, 33.4042892, -135.4551544, 135.4919128
38: -86.7652588, 50.6361198, -86.9485626, 50.6175919, -137.3828430, 137.5846863
39: -97.7150040, 43.9318314, -97.7938690, 43.9507637, -141.6657715, 141.7257080
40: -78.4429169, 34.3599663, -78.4552155, 34.4126740, -112.8555908, 112.8151779
41: -64.5526276, 40.7173347, -64.6098785, 40.7656326, -105.3182526, 105.3272095
42: -48.4880524, 36.0658112, -48.5209122, 36.1229248, -84.6109772, 84.5867233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=434, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1773

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9846796
time: 90.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0139391
time: 91.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -68.9813385, 38.8465958, -68.6670151, 38.8509521, -107.8322906, 107.5136032
1: -36.5028000, 36.5846176, -36.2485580, 36.6086884, -73.1114883, 72.8331757
2: -32.2319717, 38.6415062, -32.0240021, 38.6652222, -70.8971786, 70.6655121
3: -35.7461395, 43.0312233, -35.5591507, 43.0376205, -78.7837601, 78.5903702
4: -41.7559547, 40.2983360, -41.5106964, 40.3044434, -82.0603943, 81.8090286
5: -37.2261467, 42.4649048, -37.0481987, 42.4840851, -79.7102280, 79.5131073
6: -62.4080048, 37.6210175, -62.3922844, 37.4867744, -99.8947754, 100.0133057
7: -44.5082893, 40.4767418, -44.2374306, 40.5219879, -85.0302734, 84.7141724
8: -50.1664772, 46.6380768, -49.8393860, 46.6390266, -96.8054962, 96.4774628
9: -41.0033646, 43.9073219, -40.8015137, 43.9336777, -84.9370270, 84.7088318
10: -63.3472099, 58.4887695, -63.1145325, 58.4863968, -121.8336029, 121.6033020
11: -59.4517860, 33.7355652, -59.3520737, 33.6447639, -93.0965424, 93.0876389
12: -60.8016701, 42.4962921, -60.7608604, 42.3512039, -103.1528702, 103.2571564
13: -65.4986267, 61.3911285, -65.3638763, 61.3525925, -126.8512192, 126.7550049
14: -99.5565720, 46.6982117, -99.3480835, 46.7127342, -146.2693024, 146.0462952
15: -47.9590225, 43.1147232, -47.7873459, 43.0794067, -91.0384216, 90.9020691
16: -62.7443199, 45.8168564, -62.5141907, 45.8531532, -108.5974731, 108.3310471
17: -96.5188293, 43.8554955, -96.2793121, 43.8347626, -140.3535767, 140.1347961
18: -59.4582596, 48.0364113, -59.4255371, 47.8719292, -107.3301849, 107.4619370
19: -48.6369705, 28.0651665, -48.6219788, 27.8924904, -76.5294647, 76.6871490
20: -46.5669632, 32.2271042, -46.5690384, 32.0679169, -78.6348801, 78.7961426
21: -58.3528252, 32.8616829, -58.3015480, 32.7002411, -91.0530701, 91.1632309
22: -61.0381203, 34.7434082, -61.0400696, 34.5672455, -95.6053619, 95.7834778
23: -47.1873589, 35.7105522, -47.1761894, 35.5764236, -82.7637711, 82.8867416
24: -57.4254303, 34.0922318, -57.4182701, 33.9752731, -91.4006805, 91.5104904
25: -51.5729790, 37.5060272, -51.5826149, 37.3085632, -88.8815384, 89.0886383
26: -70.0380783, 50.3745193, -70.0567474, 50.1784363, -120.2165146, 120.4312668
27: -57.0153542, 38.9692535, -56.9926414, 38.8379974, -95.8533478, 95.9618912
28: -47.6476822, 39.4171906, -47.6674538, 39.2157745, -86.8634491, 87.0846405
29: -60.2390404, 30.6658039, -60.2095146, 30.5227737, -90.7618103, 90.8753128
30: -58.4614525, 40.6112976, -58.4263878, 40.5220566, -98.9835052, 99.0376740
31: -59.5314255, 34.5347824, -59.5074615, 34.3087730, -93.8401947, 94.0422440
32: -61.0788765, 35.9009819, -61.0627670, 35.7756882, -96.8545609, 96.9637451
33: -86.4709015, 46.4771538, -86.4612732, 46.3113747, -132.7822723, 132.9384308
34: -75.1287537, 32.0720596, -75.1610870, 31.8786049, -107.0073547, 107.2331390
35: -70.9119720, 35.3430977, -70.9447098, 35.1349335, -106.0469055, 106.2878113
36: -71.8293304, 38.0430069, -71.8759308, 37.7776375, -109.6069489, 109.9189377
37: -102.1936264, 33.5062103, -102.1881714, 33.4055862, -135.5992126, 135.6943817
38: -86.9363403, 50.9728050, -86.9975967, 50.6212997, -137.5576477, 137.9703979
39: -97.8446808, 44.0990944, -97.8193283, 43.9526863, -141.7973633, 141.9184265
40: -78.5309448, 34.4927216, -78.4659958, 34.4141617, -112.9451065, 112.9587173
41: -64.6399994, 40.8646011, -64.6284790, 40.7667885, -105.4067841, 105.4930801
42: -48.5447083, 36.1995354, -48.5308800, 36.1240311, -84.6687317, 84.7304153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=434, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1773

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9843979
time: 119.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0525436
time: 91.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -69.2380447, 38.8934021, -68.4611816, 38.7577019, -107.9957275, 107.3545761
1: -36.6817551, 36.6270752, -36.1453934, 36.5375900, -73.2193451, 72.7724686
2: -32.3680725, 38.6902428, -31.9090652, 38.5943451, -70.9624176, 70.5993042
3: -35.8758163, 43.0731583, -35.4546127, 42.9426193, -78.8184280, 78.5277710
4: -41.9350090, 40.3319397, -41.3833237, 40.2421722, -82.1771851, 81.7152634
5: -37.3776283, 42.5176239, -36.9545517, 42.3889389, -79.7665634, 79.4721756
6: -62.4528656, 37.6924019, -62.3026581, 37.2455635, -99.6984100, 99.9950562
7: -44.7438469, 40.5482788, -44.1790733, 40.4556847, -85.1995316, 84.7273560
8: -50.3796234, 46.6927147, -49.6947327, 46.5615692, -96.9411926, 96.3874435
9: -41.1735382, 43.9709473, -40.6847496, 43.8587799, -85.0323105, 84.6557007
10: -63.5208435, 58.5529594, -62.9683037, 58.3828735, -121.9037170, 121.5212479
11: -59.5597687, 33.7517624, -59.2687836, 33.4851112, -93.0448761, 93.0205460
12: -60.8295975, 42.7068863, -60.6738281, 42.2983551, -103.1279449, 103.3807144
13: -65.6077881, 61.4455910, -65.1661835, 61.2273216, -126.8351135, 126.6117630
14: -99.6952286, 46.7520599, -99.0481949, 46.5903931, -146.2856140, 145.8002472
15: -48.0125999, 43.2071037, -47.5765610, 43.0256157, -91.0382004, 90.7836609
16: -62.9465179, 45.8961182, -62.4124069, 45.7637329, -108.7102203, 108.3085251
17: -96.6869125, 43.9095955, -96.1045456, 43.7021027, -140.3890076, 140.0141449
18: -59.5311089, 48.1472969, -59.3398056, 47.7396774, -107.2707825, 107.4871063
19: -48.6893997, 28.1279869, -48.4885788, 27.7295818, -76.4189835, 76.6165619
20: -46.6132812, 32.2771606, -46.4285240, 31.8785019, -78.4917831, 78.7056885
21: -58.4158974, 32.8834190, -58.1507416, 32.4673500, -90.8832397, 91.0341644
22: -61.1340523, 34.8486557, -60.8978310, 34.4411736, -95.5752258, 95.7464905
23: -47.2314453, 35.7905960, -47.0616417, 35.4520988, -82.6835403, 82.8522339
24: -57.4845505, 34.1490974, -57.2778511, 33.8474159, -91.3319702, 91.4269485
25: -51.6423149, 37.5987587, -51.4628372, 37.1638565, -88.8061676, 89.0615921
26: -70.1414337, 50.5166168, -69.9159927, 50.0000343, -120.1414642, 120.4326096
27: -57.0772362, 39.0306320, -56.8471375, 38.6559677, -95.7332001, 95.8777695
28: -47.7049828, 39.5283813, -47.5443039, 39.0643082, -86.7692871, 87.0726776
29: -60.3292198, 30.7611561, -60.0788383, 30.4233780, -90.7525940, 90.8399963
30: -58.5386314, 40.5941696, -58.3020096, 40.3050804, -98.8437119, 98.8961792
31: -59.6032104, 34.6537132, -59.3852844, 34.1547928, -93.7579880, 94.0390015
32: -61.1250801, 36.0021744, -60.9682732, 35.6304245, -96.7554932, 96.9704437
33: -86.5596161, 46.6259384, -86.3349762, 46.1815224, -132.7411346, 132.9609070
34: -75.2180786, 32.2457085, -75.0700531, 31.7754955, -106.9935760, 107.3157578
35: -71.0130463, 35.5316048, -70.8580017, 35.0680656, -106.0811005, 106.3896027
36: -71.9315948, 38.2502708, -71.7673492, 37.6754456, -109.6070404, 110.0176086
37: -102.2989120, 33.6659775, -102.0211945, 33.3470802, -135.6459961, 135.6871643
38: -87.0787506, 51.2604561, -86.8629456, 50.5144768, -137.5932007, 138.1233978
39: -97.9313736, 44.2254829, -97.6619186, 43.8933487, -141.8247223, 141.8873901
40: -78.5991821, 34.6410484, -78.3701172, 34.3933792, -112.9925461, 113.0111618
41: -64.7053223, 40.9571800, -64.5345917, 40.6071854, -105.3125076, 105.4917755
42: -48.5783386, 36.2966843, -48.4705009, 36.0062790, -84.5846176, 84.7671814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=434, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1773

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9881577
time: 86.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0185597
time: 88.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -69.1285629, 38.8661575, -68.8789291, 38.8622360, -107.9907990, 107.7450867
1: -36.5793839, 36.5862236, -36.3890228, 36.6017990, -73.1811829, 72.9752502
2: -32.2992592, 38.6432114, -32.1438065, 38.6582336, -70.9574890, 70.7870178
3: -35.8038483, 43.0398216, -35.6709328, 43.0421410, -78.8459930, 78.7107544
4: -41.8383713, 40.2983627, -41.6589584, 40.3055344, -82.1439056, 81.9573212
5: -37.2941818, 42.4766312, -37.1699257, 42.4840775, -79.7782593, 79.6465530
6: -62.4278793, 37.6864548, -62.4128151, 37.5989876, -100.0268478, 100.0992584
7: -44.5999641, 40.4853134, -44.4001808, 40.5126038, -85.1125641, 84.8854904
8: -50.2434196, 46.6345749, -50.0137291, 46.6398125, -96.8832092, 96.6483002
9: -41.0941124, 43.9167061, -40.9384422, 43.9314842, -85.0255966, 84.8551483
10: -63.4320259, 58.5053215, -63.2602158, 58.5016785, -121.9337006, 121.7655334
11: -59.4850235, 33.7623138, -59.4116936, 33.6993599, -93.1843872, 93.1740112
12: -60.8246574, 42.5647087, -60.7873230, 42.4776421, -103.3022919, 103.3520355
13: -65.6096344, 61.4151459, -65.4854431, 61.3836365, -126.9932709, 126.9005890
14: -99.6603165, 46.6992188, -99.5117569, 46.7130890, -146.3734131, 146.2109680
15: -47.9724007, 43.1934853, -47.8726616, 43.1392212, -91.1116028, 91.0661469
16: -62.8398170, 45.8352890, -62.6591492, 45.8524323, -108.6922455, 108.4944382
17: -96.5993652, 43.8825150, -96.4206467, 43.8593826, -140.4587402, 140.3031616
18: -59.4792862, 48.1048622, -59.4507523, 47.9792252, -107.4585114, 107.5556030
19: -48.6422615, 28.1161537, -48.6328354, 27.9861374, -76.6283875, 76.7489929
20: -46.5756874, 32.2733803, -46.5771408, 32.1568756, -78.7325592, 78.8505249
21: -58.3637199, 32.9031296, -58.3313065, 32.7861710, -91.1498871, 91.2344360
22: -61.0559769, 34.8188286, -61.0523300, 34.6717072, -95.7276688, 95.8711548
23: -47.1956863, 35.7701912, -47.1871758, 35.6644440, -82.8601303, 82.9573669
24: -57.4494476, 34.1554337, -57.4376755, 34.0507660, -91.5002060, 91.5931015
25: -51.5828667, 37.5626373, -51.5884857, 37.4092255, -88.9920807, 89.1511230
26: -70.0704422, 50.4832497, -70.0683899, 50.3246307, -120.3950729, 120.5516357
27: -57.0408173, 39.0595627, -57.0174103, 38.9314041, -95.9722137, 96.0769653
28: -47.6601410, 39.4926720, -47.6690788, 39.3323517, -86.9924927, 87.1617432
29: -60.2589340, 30.7390423, -60.2340508, 30.6120148, -90.8709412, 90.9730911
30: -58.5013428, 40.6417236, -58.4726562, 40.5697632, -99.0710907, 99.1143799
31: -59.5437698, 34.6057510, -59.5266190, 34.4324341, -93.9762039, 94.1323700
32: -61.0972862, 35.9570084, -61.0824432, 35.8725662, -96.9698486, 97.0394516
33: -86.4978333, 46.5402603, -86.4822693, 46.4370117, -132.9348450, 133.0225220
34: -75.1550140, 32.1694489, -75.1633453, 32.0173149, -107.1723328, 107.3327789
35: -70.9338226, 35.4241638, -70.9469147, 35.2713165, -106.2051392, 106.3710785
36: -71.8400269, 38.1300163, -71.8667450, 37.9371796, -109.7771912, 109.9967575
37: -102.2159271, 33.5764046, -102.2046432, 33.5205994, -135.7365265, 135.7810516
38: -86.9476929, 51.0728111, -86.9847565, 50.8270569, -137.7747498, 138.0575714
39: -97.8706970, 44.1243858, -97.8454819, 44.0428848, -141.9135742, 141.9698486
40: -78.5625000, 34.5647240, -78.5025177, 34.5087738, -113.0712738, 113.0672455
41: -64.6556778, 40.9385605, -64.6407623, 40.8705444, -105.5262146, 105.5793228
42: -48.5571518, 36.2637405, -48.5437698, 36.2165604, -84.7737122, 84.8075027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=434, inp2_unstable=435, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=566, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1480

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1773

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9516092
time: 86.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0189099
time: 91.33 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 180.12 seconds
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9986665
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0148196
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0027793
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9808316
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9843979
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0146498
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9846796
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0139391
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9843979
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0525436
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9881577
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0185597
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 50.9516092
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 180.12
Output dim: 1, lower bound: -51.0081817, upper bound: 51.0189099
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 180.12
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0571064

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 113.30 + 7177.21 = 7290.51 seconds
