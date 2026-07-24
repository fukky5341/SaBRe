## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 7200 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.84 + 117.34 = 120.17 seconds
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0645988, upper bound: 51.0301099
time: 99.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0645988, upper bound: 51.0645987
time: 123.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 223.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 223.13
Output dim: 1, lower bound: -51.0645988, upper bound: 51.0301099
IS_A2, status: Status.UNKNOWN, split count: 1, time: 223.13
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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0114332
time: 190.58 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0267565
time: 98.45 seconds

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

Time for backsubstitution: 2.19 seconds

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
time: 113.03 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0635915
time: 101.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 217.21 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 217.21
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0114332
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 217.21
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0267565
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 217.21
Output dim: 1, lower bound: -51.0635916, upper bound: 51.0278149
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 217.21
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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0194119, upper bound: 51.0047181
time: 107.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0106063
time: 93.96 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0194699, upper bound: 51.0202940
time: 106.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0259217
time: 112.99 seconds

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0194119, upper bound: 51.0207393
time: 92.24 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0264228
time: 98.73 seconds

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

Time for backsubstitution: 2.24 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0194699, upper bound: 51.0567540
time: 86.11 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0622010
time: 175.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 263.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 263.72
Output dim: 1, lower bound: -51.0194119, upper bound: 51.0047181
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 263.72
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0106063
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 263.72
Output dim: 1, lower bound: -51.0194699, upper bound: 51.0202940
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 263.72
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0259217
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 263.72
Output dim: 1, lower bound: -51.0194119, upper bound: 51.0207393
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 263.72
Output dim: 1, lower bound: -51.0622010, upper bound: 51.0264228
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 263.72
Output dim: 1, lower bound: -51.0194699, upper bound: 51.0567540
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 263.72
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

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9646235
time: 99.22 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0025596
time: 99.40 seconds

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

Time for backsubstitution: 2.20 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9704890
time: 101.70 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0084360
time: 99.05 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 50.9802215
time: 202.85 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 50.9802215
time: 774.85 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0569152, upper bound: 50.9857949
time: 103.48 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0237466
time: 99.76 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9806372
time: 122.08 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0185907
time: 113.28 seconds

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

Time for backsubstitution: 2.20 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9863094
time: 93.99 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0242543
time: 105.76 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0167462
time: 107.14 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0546244
time: 94.95 seconds

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

Time for backsubstitution: 2.25 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0221640
time: 120.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0600534
time: 121.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 244.89 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9646235
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0025596
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9704890
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0084360
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141910, upper bound: 50.9802215
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141910, upper bound: 50.9802215
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0569152, upper bound: 50.9857949
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0237466
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9806372
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0185907
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9863094
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0242543
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0167462
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0546244
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0221640
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 244.89
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0600534

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -67.9368744, 38.6802826, -67.7169800, 38.6101341, -106.5470123, 106.3972626
1: -35.8041840, 36.4621887, -35.6753769, 36.4257202, -72.2299042, 72.1375504
2: -31.6390324, 38.4982376, -31.5154743, 38.4698219, -70.1088562, 70.0137100
3: -35.2035332, 42.8672714, -35.0909805, 42.8068008, -78.0103302, 77.9582520
4: -41.0276108, 40.1616516, -40.8700905, 40.1287231, -81.1563339, 81.0317383
5: -36.6693115, 42.2772064, -36.5572166, 42.2210197, -78.8903351, 78.8344193
6: -62.2460785, 37.0544891, -62.1729050, 36.8850861, -99.1311646, 99.2273941
7: -43.7709541, 40.3042831, -43.6595955, 40.2916489, -84.0626068, 83.9638824
8: -49.2952766, 46.4604340, -49.1387596, 46.4283676, -95.7236481, 95.5991898
9: -40.3442078, 43.7248154, -40.2085190, 43.6942902, -84.0384827, 83.9333267
10: -62.6239319, 58.2479858, -62.4779510, 58.1870193, -120.8109512, 120.7259293
11: -59.1253242, 33.3848114, -59.0320244, 33.2558212, -92.3811493, 92.4168243
12: -60.5897865, 41.9754333, -60.5281219, 41.9120865, -102.5018616, 102.5035553
13: -64.8846054, 61.0948105, -64.6877594, 60.9746323, -125.8592224, 125.7825699
14: -98.6999435, 46.5097351, -98.4453049, 46.4300308, -145.1299744, 144.9550476
15: -47.4124985, 42.8622208, -47.2638702, 42.7897263, -90.2022247, 90.1260910
16: -62.0216064, 45.5963211, -61.8925552, 45.5532227, -107.5748291, 107.4888687
17: -95.7468796, 43.6505013, -95.5537415, 43.5444794, -139.2913513, 139.2042389
18: -59.1981544, 47.5170746, -59.1471519, 47.4044189, -106.6025696, 106.6642303
19: -48.3849525, 27.5546360, -48.2935181, 27.4040833, -75.7890320, 75.8481445
20: -46.3604546, 31.7186546, -46.2640533, 31.5681725, -77.9286270, 77.9827118
21: -58.0409851, 32.3396072, -57.9234276, 32.1593170, -90.2003021, 90.2630310
22: -60.7039146, 34.2173195, -60.6232033, 34.0936089, -94.7975159, 94.8405228
23: -46.9720726, 35.2474594, -46.8789062, 35.1096802, -82.0817566, 82.1263657
24: -57.1678581, 33.6899338, -57.0565987, 33.5652237, -90.7330780, 90.7465286
25: -51.3049583, 36.9529648, -51.2540474, 36.8168716, -88.1218262, 88.2070160
26: -69.7462845, 49.6889915, -69.6438370, 49.5168037, -119.2630920, 119.3328247
27: -56.7196884, 38.4703369, -56.6011467, 38.3038788, -95.0235672, 95.0714874
28: -47.4195023, 38.8030243, -47.3410645, 38.6452293, -86.0647202, 86.1440887
29: -59.9011383, 30.2146511, -59.8072433, 30.0979233, -89.9990540, 90.0218964
30: -58.2114983, 40.2445984, -58.1079369, 40.0973892, -98.3088837, 98.3525314
31: -59.2202606, 33.8961525, -59.1488152, 33.7450676, -92.9653320, 93.0449600
32: -60.8775406, 35.4440155, -60.8154144, 35.3312263, -96.2087708, 96.2594299
33: -86.1692963, 45.9065475, -86.0875244, 45.7928238, -131.9621124, 131.9940796
34: -74.8815536, 31.4517403, -74.8485413, 31.3371162, -106.2186737, 106.3002777
35: -70.6490173, 34.7492714, -70.6216278, 34.6618538, -105.3108673, 105.3708954
36: -71.5756226, 37.3178406, -71.5376816, 37.2011108, -108.7767334, 108.8555145
37: -101.8583069, 33.0802765, -101.7419052, 33.0087318, -134.8670349, 134.8221741
38: -86.5958710, 50.0271950, -86.5568542, 49.8966331, -136.4925079, 136.5840454
39: -97.4816284, 43.7123299, -97.3844376, 43.6623001, -141.1439209, 141.0967712
40: -78.2270966, 34.1685562, -78.1542358, 34.1363678, -112.3634644, 112.3227844
41: -64.4206696, 40.3949661, -64.3472290, 40.2484283, -104.6690979, 104.7421875
42: -48.3874512, 35.7992401, -48.3550949, 35.6860504, -84.0735016, 84.1543274

Time for backsubstitution: 2.21 seconds

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
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1757
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
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1665
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
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1731
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
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
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
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1705
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
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 593
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
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1578
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
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 520
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
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 535
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
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 956
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
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1712
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
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 527
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
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 981
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
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1368
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
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1653

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9338574
time: 139.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9624553
time: 102.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -68.1377335, 38.7069016, -68.1391754, 38.7256889, -106.8634109, 106.8460770
1: -35.9227142, 36.4763489, -35.9209099, 36.5066795, -72.4293976, 72.3972626
2: -31.7535896, 38.5138550, -31.7508259, 38.5524216, -70.3060150, 70.2646790
3: -35.3088760, 42.8965340, -35.3082504, 42.9213409, -78.2302170, 78.2047882
4: -41.1609650, 40.1834259, -41.1467590, 40.2062378, -81.3672028, 81.3301849
5: -36.7739944, 42.3054314, -36.7734261, 42.3345261, -79.1085205, 79.0788574
6: -62.2786751, 37.2252159, -62.2910042, 37.2404327, -99.5191040, 99.5162201
7: -43.8761902, 40.3211060, -43.8819542, 40.3730392, -84.2492142, 84.2030563
8: -49.4505005, 46.4821243, -49.4600563, 46.5269318, -95.9774323, 95.9421844
9: -40.4674911, 43.7485733, -40.4648209, 43.7876282, -84.2551193, 84.2133942
10: -62.7663345, 58.2920914, -62.7739563, 58.3240128, -121.0903473, 121.0660477
11: -59.1633835, 33.4881592, -59.1807137, 33.4713440, -92.6347275, 92.6688614
12: -60.6429405, 42.0517616, -60.6479530, 42.0965157, -102.7394562, 102.6997147
13: -65.0363922, 61.1421928, -65.0094757, 61.1404419, -126.1768341, 126.1516724
14: -98.9184494, 46.5438614, -98.9124069, 46.5746231, -145.4930420, 145.4562683
15: -47.5563164, 42.9056206, -47.5643959, 42.9065933, -90.4629059, 90.4700165
16: -62.1367416, 45.6272507, -62.1439857, 45.6657410, -107.8024826, 107.7712402
17: -95.8925552, 43.6855927, -95.8748169, 43.7100525, -139.6026001, 139.5604095
18: -59.2366562, 47.6332436, -59.2716904, 47.6458435, -106.8824921, 106.9049301
19: -48.4214630, 27.6806068, -48.4544601, 27.6596985, -76.0811615, 76.1350708
20: -46.3934288, 31.8556023, -46.4279594, 31.8476639, -78.2410889, 78.2835541
21: -58.0881844, 32.4971390, -58.1170998, 32.4789124, -90.5670776, 90.6142426
22: -60.7380753, 34.3299370, -60.8018684, 34.3252869, -95.0633621, 95.1318054
23: -47.0053406, 35.3519096, -47.0179100, 35.3227844, -82.3281250, 82.3698120
24: -57.2001572, 33.7905273, -57.2290154, 33.7691612, -90.9693146, 91.0195389
25: -51.3393135, 37.0722275, -51.4017105, 37.0637131, -88.4030304, 88.4739380
26: -69.7850189, 49.8501015, -69.8213348, 49.8421478, -119.6271591, 119.6714325
27: -56.7575722, 38.6065903, -56.7826004, 38.5806313, -95.3381958, 95.3891907
28: -47.4486389, 38.9365807, -47.4844627, 38.9143600, -86.3629990, 86.4210434
29: -59.9346123, 30.3057899, -59.9798393, 30.2872143, -90.2218246, 90.2856293
30: -58.2514458, 40.3744850, -58.2837448, 40.3657799, -98.6172256, 98.6582336
31: -59.2616730, 34.0321426, -59.3093147, 34.0226288, -93.2843018, 93.3414612
32: -60.9190788, 35.5556908, -60.9419441, 35.5758934, -96.4949646, 96.4976349
33: -86.2256927, 46.0293655, -86.2592926, 46.0512047, -132.2768860, 132.2886353
34: -74.9176865, 31.5702400, -74.9677582, 31.5805893, -106.4982758, 106.5379944
35: -70.6826477, 34.8470993, -70.7406235, 34.8664856, -105.5491333, 105.5877151
36: -71.6037292, 37.4455566, -71.6720810, 37.4645996, -109.0683289, 109.1176376
37: -101.9186020, 33.1628685, -101.9562683, 33.1832428, -135.1018372, 135.1191406
38: -86.6360321, 50.1769257, -86.7278366, 50.2127457, -136.8487854, 136.9047546
39: -97.5496750, 43.7793579, -97.5937271, 43.8134995, -141.3631744, 141.3730774
40: -78.2781906, 34.2210312, -78.2971344, 34.2528305, -112.5310135, 112.5181580
41: -64.4579315, 40.5218430, -64.4722290, 40.5121880, -104.9701233, 104.9940720
42: -48.4228439, 35.8973694, -48.4382057, 35.8964653, -84.3193054, 84.3355637

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
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
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1701
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
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1714
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
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1557
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
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1670
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
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1463
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
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 593
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
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1578
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
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 964
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
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 520
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
type: A, layer: 1, pos: 1755
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
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 956
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
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 835
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 965
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
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1368
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
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9718351
time: 112.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0004006
time: 121.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -68.3970337, 38.7541885, -67.9331970, 38.6324692, -107.0295029, 106.6873856
1: -36.1034584, 36.5187607, -35.8178215, 36.4355927, -72.5390472, 72.3365784
2: -31.8906307, 38.5627289, -31.6358185, 38.4815216, -70.3721466, 70.1985474
3: -35.4389305, 42.9385376, -35.2035980, 42.8262291, -78.2651443, 78.1421356
4: -41.3413963, 40.2172852, -41.0193253, 40.1439896, -81.4853821, 81.2366028
5: -36.9259109, 42.3584976, -36.6798401, 42.2393417, -79.1652527, 79.0383377
6: -62.3259544, 37.2980385, -62.2007866, 36.9993744, -99.3253326, 99.4988251
7: -44.1122818, 40.3931847, -43.8237038, 40.3067627, -84.4190369, 84.2168808
8: -49.6645393, 46.5371971, -49.3149033, 46.4495201, -96.1140594, 95.8520889
9: -40.6387634, 43.8128052, -40.3479691, 43.7127304, -84.3514938, 84.1607742
10: -62.9405670, 58.3573990, -62.6271324, 58.2206230, -121.1611938, 120.9845276
11: -59.2640724, 33.5053635, -59.0961494, 33.3114777, -92.5755463, 92.6015167
12: -60.6738892, 42.2606621, -60.5602303, 42.0432892, -102.7171707, 102.8208923
13: -65.1489792, 61.1978836, -64.8116760, 61.0152588, -126.1642151, 126.0095596
14: -99.0606003, 46.5998154, -98.6135712, 46.4522705, -145.5128784, 145.2133789
15: -47.6103210, 42.9977379, -47.3535690, 42.8525162, -90.4628372, 90.3513031
16: -62.3394966, 45.7077103, -62.0421638, 45.5763435, -107.9158401, 107.7498779
17: -96.0651169, 43.7398186, -95.7027893, 43.5773697, -139.6424866, 139.4426117
18: -59.3094368, 47.7456512, -59.1861458, 47.5130310, -106.8224640, 106.9317932
19: -48.4740486, 27.7451782, -48.3211365, 27.4964237, -75.9704742, 76.0663147
20: -46.4396362, 31.9072685, -46.2874107, 31.6578922, -78.0975266, 78.1946793
21: -58.1514816, 32.5202637, -57.9665794, 32.2454262, -90.3968887, 90.4868393
22: -60.8362198, 34.4353638, -60.6599846, 34.1984558, -95.0346756, 95.0953522
23: -47.0477905, 35.4333305, -46.9034462, 35.1979904, -82.2457809, 82.3367767
24: -57.2593842, 33.8480835, -57.0885086, 33.6404076, -90.8997879, 90.9365921
25: -51.4089394, 37.1652756, -51.2818375, 36.9182053, -88.3271484, 88.4471130
26: -69.8881378, 49.9942284, -69.6805649, 49.6633759, -119.5515137, 119.6747894
27: -56.8211441, 38.6690903, -56.6375160, 38.3977852, -95.2189255, 95.3065872
28: -47.5054550, 39.0493355, -47.3612633, 38.7623138, -86.2677689, 86.4105988
29: -60.0273018, 30.4017639, -59.8496971, 30.1873302, -90.2146301, 90.2514572
30: -58.3277397, 40.3569870, -58.1588593, 40.1476135, -98.4753571, 98.5158310
31: -59.3328819, 34.1528168, -59.1869392, 33.8682022, -93.2010727, 93.3397522
32: -60.9674988, 35.6568222, -60.8475227, 35.4305000, -96.3979797, 96.5043488
33: -86.3155746, 46.1782608, -86.1330719, 45.9210587, -132.2366333, 132.3113403
34: -75.0077286, 31.7451706, -74.8767853, 31.4777260, -106.4854507, 106.6219559
35: -70.7845459, 35.0363846, -70.6539230, 34.7994728, -105.5839996, 105.6903076
36: -71.7072906, 37.6530418, -71.5636292, 37.3621254, -109.0694122, 109.2166748
37: -102.0250931, 33.3219147, -101.7892914, 33.1244888, -135.1495819, 135.1112061
38: -86.7795410, 50.4646187, -86.5932007, 50.1056786, -136.8852234, 137.0578156
39: -97.6388855, 43.9058533, -97.4365997, 43.7543869, -141.3932800, 141.3424530
40: -78.3463287, 34.3739586, -78.2007751, 34.2333870, -112.5797043, 112.5747375
41: -64.5250244, 40.6157188, -64.3783493, 40.3526573, -104.8776855, 104.9940643
42: -48.4583778, 35.9970779, -48.3779144, 35.7792053, -84.2375793, 84.3749847

Time for backsubstitution: 2.23 seconds

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
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1665
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
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1731
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
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1679
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
type: A, layer: 1, pos: 1463
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1766
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
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 956
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1434
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

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9391261
time: 99.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9662864
time: 92.49 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9771141
time: 100.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0042155
time: 106.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -68.1822052, 38.7093239, -68.2296982, 38.7071953, -106.8894043, 106.9390259
1: -35.9583664, 36.4754372, -35.9963799, 36.4947166, -72.4530792, 72.4718170
2: -31.7697964, 38.5126877, -31.7855453, 38.5440445, -70.3138428, 70.2982330
3: -35.3229370, 42.8896141, -35.3387413, 42.8923721, -78.2153015, 78.2283478
4: -41.2005539, 40.1794891, -41.2300720, 40.1985855, -81.3991241, 81.4095535
5: -36.8006897, 42.3006287, -36.8300018, 42.3286171, -79.1293030, 79.1306305
6: -62.2840157, 37.1695404, -62.2673187, 37.1248512, -99.4088669, 99.4368439
7: -43.9414444, 40.3208237, -44.0110321, 40.3906097, -84.3320541, 84.3318558
8: -49.4728317, 46.4851303, -49.5100327, 46.5055580, -95.9783936, 95.9951630
9: -40.4996681, 43.7453461, -40.5352936, 43.7969589, -84.2966309, 84.2806396
10: -62.7804604, 58.2872353, -62.8082085, 58.3150558, -121.0955200, 121.0954437
11: -59.1912766, 33.4635429, -59.1822281, 33.4245300, -92.6158066, 92.6457672
12: -60.6362343, 42.0872841, -60.6338272, 42.1521416, -102.7883759, 102.7211075
13: -65.0535278, 61.1435432, -65.0384827, 61.1412239, -126.1947479, 126.1820221
14: -98.9014282, 46.5361252, -98.8706589, 46.5364876, -145.4379120, 145.4067841
15: -47.5095520, 42.9398117, -47.4752769, 42.9527397, -90.4622879, 90.4150848
16: -62.1892700, 45.6216660, -62.2454720, 45.6892853, -107.8785553, 107.8671417
17: -95.9349823, 43.6860695, -95.9493942, 43.6440659, -139.5790405, 139.6354675
18: -59.2435303, 47.6213989, -59.2688141, 47.6242142, -106.8677292, 106.8901978
19: -48.4179955, 27.6687717, -48.4125443, 27.6368866, -76.0548859, 76.0813141
20: -46.3897057, 31.8242722, -46.3642921, 31.7857952, -78.1754990, 78.1885529
21: -58.0921631, 32.4462624, -58.0654106, 32.3792419, -90.4714050, 90.5116730
22: -60.7464485, 34.3332710, -60.7866325, 34.3310051, -95.0774536, 95.1198959
23: -47.0016098, 35.3684578, -46.9957924, 35.3610992, -82.3627014, 82.3642502
24: -57.2098045, 33.7883911, -57.1923409, 33.7698936, -90.9796982, 90.9807281
25: -51.3370399, 37.0690918, -51.3727493, 37.0571976, -88.3942413, 88.4418411
26: -69.7956696, 49.8526230, -69.8263092, 49.8516083, -119.6472778, 119.6789246
27: -56.7642860, 38.5938110, -56.7537231, 38.5594673, -95.3237457, 95.3475342
28: -47.4447517, 38.9483871, -47.4653015, 38.9448090, -86.3895569, 86.4136887
29: -59.9500732, 30.3267937, -59.9721832, 30.3298512, -90.2799149, 90.2989807
30: -58.2617073, 40.3130417, -58.2257729, 40.2461395, -98.5078430, 98.5388107
31: -59.2639313, 34.0331879, -59.2879295, 34.0283318, -93.2922668, 93.3211212
32: -60.9198685, 35.5373497, -60.9193764, 35.5250435, -96.4449158, 96.4567261
33: -86.2284012, 46.0223999, -86.2572556, 46.0434875, -132.2718811, 132.2796631
34: -74.9158783, 31.5948792, -74.9927673, 31.6307182, -106.5466003, 106.5876465
35: -70.6901932, 34.8753090, -70.7725525, 34.9255219, -105.6156921, 105.6478577
36: -71.6078720, 37.4671326, -71.6817932, 37.5107269, -109.1185989, 109.1489258
37: -101.9177170, 33.1858215, -101.9353943, 33.2270432, -135.1447601, 135.1212158
38: -86.6373444, 50.2199173, -86.7501831, 50.2957115, -136.9330444, 136.9700928
39: -97.5538406, 43.7724228, -97.5713043, 43.7951660, -141.3489990, 141.3437195
40: -78.2860031, 34.2453232, -78.2987823, 34.2952652, -112.5812683, 112.5441055
41: -64.4631500, 40.5186920, -64.4797440, 40.5025940, -104.9657440, 104.9984283
42: -48.4172401, 35.9083710, -48.4357109, 35.9124908, -84.3297272, 84.3440857

Time for backsubstitution: 2.25 seconds

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
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1701
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
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1717
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
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1731
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
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1575
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
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1463
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
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1622
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
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 593
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
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 520
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
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 956
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1434
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
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9507814
time: 107.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9780564
time: 1690.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -68.3830414, 38.7359314, -68.6520004, 38.8225937, -107.2056351, 107.3879318
1: -36.0769043, 36.4895821, -36.2419968, 36.5756416, -72.6525421, 72.7315826
2: -31.8843956, 38.5282822, -32.0209084, 38.6265945, -70.5109863, 70.5491943
3: -35.4282837, 42.9188461, -35.5560799, 43.0068626, -78.4351425, 78.4749298
4: -41.3339500, 40.2012405, -41.5069351, 40.2760468, -81.6100006, 81.7081757
5: -36.9054108, 42.3288231, -37.0463219, 42.4420204, -79.3474274, 79.3751373
6: -62.3166389, 37.3403206, -62.3854065, 37.4803162, -99.7969513, 99.7257233
7: -44.0466766, 40.3376122, -44.2334137, 40.4719429, -84.5186157, 84.5710297
8: -49.6280861, 46.5068588, -49.8314323, 46.6041679, -96.2322540, 96.3382797
9: -40.6229935, 43.7690811, -40.7917709, 43.8902283, -84.5132141, 84.5608521
10: -62.9229012, 58.3312607, -63.1043205, 58.4518242, -121.3747253, 121.4355698
11: -59.2293053, 33.5669327, -59.3310013, 33.6402206, -92.8695221, 92.8979340
12: -60.6894341, 42.1637840, -60.7538834, 42.3369484, -103.0263824, 102.9176636
13: -65.2053375, 61.1908951, -65.3603210, 61.3066483, -126.5119858, 126.5512161
14: -99.1199417, 46.5701904, -99.3377914, 46.6809502, -145.8008881, 145.9079742
15: -47.6535492, 42.9832268, -47.7761421, 43.0697021, -90.7232513, 90.7593613
16: -62.3044205, 45.6525955, -62.4969864, 45.8016586, -108.1060791, 108.1495819
17: -96.0806427, 43.7211533, -96.2705383, 43.8095512, -139.8901825, 139.9916840
18: -59.2820053, 47.7375755, -59.3932686, 47.8657990, -107.1477966, 107.1308441
19: -48.4544754, 27.7947235, -48.5733490, 27.8925323, -76.3470078, 76.3680725
20: -46.4226608, 31.9612083, -46.5280113, 32.0653229, -78.4879837, 78.4892197
21: -58.1393776, 32.6038284, -58.2589226, 32.6988831, -90.8382492, 90.8627472
22: -60.7806435, 34.4458466, -60.9651489, 34.5626793, -95.3433075, 95.4109955
23: -47.0348892, 35.4729347, -47.1346970, 35.5743217, -82.6092072, 82.6076279
24: -57.2421341, 33.8890114, -57.3646622, 33.9739113, -91.2160339, 91.2536697
25: -51.3713417, 37.1883926, -51.5202408, 37.3040543, -88.6753998, 88.7086334
26: -69.8343964, 50.0138016, -70.0036621, 50.1769981, -120.0113983, 120.0174637
27: -56.8022041, 38.7300873, -56.9351234, 38.8363953, -95.6385956, 95.6652069
28: -47.4738693, 39.0819588, -47.6085701, 39.2139664, -86.6878357, 86.6905289
29: -59.9835396, 30.4179363, -60.1447601, 30.5191326, -90.5026703, 90.5626984
30: -58.3016548, 40.4430008, -58.4015923, 40.5146942, -98.8163376, 98.8445892
31: -59.3053360, 34.1692238, -59.4482002, 34.3059998, -93.6113281, 93.6174240
32: -60.9613190, 35.6490097, -61.0456810, 35.7699051, -96.7312164, 96.6946869
33: -86.2847595, 46.1452980, -86.4286499, 46.3019600, -132.5867004, 132.5739441
34: -74.9519653, 31.7134018, -75.1118393, 31.8741970, -106.8261414, 106.8252411
35: -70.7237854, 34.9730797, -70.8913574, 35.1302147, -105.8540039, 105.8644333
36: -71.6359253, 37.5948486, -71.8160400, 37.7741928, -109.4101105, 109.4108810
37: -101.9780884, 33.2683296, -102.1495285, 33.4016190, -135.3796997, 135.4178619
38: -86.6773834, 50.3696632, -86.9209824, 50.6118546, -137.2892456, 137.2906342
39: -97.6217728, 43.8394623, -97.7800827, 43.9464226, -141.5681915, 141.6195374
40: -78.3370819, 34.2977142, -78.4417496, 34.4116135, -112.7486954, 112.7394638
41: -64.5003815, 40.6456146, -64.6045074, 40.7664490, -105.2668152, 105.2501144
42: -48.4526405, 36.0065575, -48.5187836, 36.1230392, -84.5756683, 84.5253372

Time for backsubstitution: 2.22 seconds

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
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
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
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1588
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
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1670
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
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1463
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
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1599
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
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 536
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
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1654
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
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1703
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
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 965
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
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 752
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
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1480

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9887322
time: 102.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0159850
time: 92.95 seconds

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

Time for backsubstitution: 2.20 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9558195
time: 113.21 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9817063
time: 97.28 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 212.82 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9338574
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9624553
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9718351
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0004006
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9391261
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9662864
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9771141
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 51.0042155
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9507814
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9780564
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9887322
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0102746, upper bound: 51.0159850
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0101446, upper bound: 50.9558195
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 212.82
Output dim: 1, lower bound: -51.0102746, upper bound: 50.9817063
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0600534, upper bound: 51.0237466
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9806372
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0185907
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0141397, upper bound: 50.9863094
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0141397, upper bound: 51.0242543
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0167462
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0546244
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0221640
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 212.82
Output dim: 1, lower bound: -51.0141910, upper bound: 51.0600534

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 120.17 + 7187.96 = 7308.13 seconds
