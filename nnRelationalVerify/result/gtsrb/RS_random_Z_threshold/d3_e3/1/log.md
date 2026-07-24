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
execution time: IAR + RelationalAnalysis = 3.00 + 127.07 = 130.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -51.0653554, upper bound: 51.0653554

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 529

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0541902, upper bound: 51.0645941
time: 101.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0645941, upper bound: 51.0541902
time: 98.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 200.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 200.19
Output dim: 1, lower bound: -51.0541902, upper bound: 51.0645941
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 200.19
Output dim: 1, lower bound: -51.0645941, upper bound: 51.0541902

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 984

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0088942, upper bound: 51.0246666
time: 92.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0142943, upper bound: 51.0192364
time: 88.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1559

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 732

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0642457, upper bound: 51.0415453
time: 116.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0519466, upper bound: 51.0538529
time: 99.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 218.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 218.56
Output dim: 1, lower bound: -51.0088942, upper bound: 51.0246666
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 218.56
Output dim: 1, lower bound: -51.0142943, upper bound: 51.0192364
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 218.56
Output dim: 1, lower bound: -51.0642457, upper bound: 51.0415453
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 218.56
Output dim: 1, lower bound: -51.0519466, upper bound: 51.0538529

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 908

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0000946, upper bound: 51.0244026
time: 88.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0086291, upper bound: 51.0158515
time: 112.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 839

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1545

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0073332, upper bound: 51.0190804
time: 99.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0141388, upper bound: 51.0122490
time: 97.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1740

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0633247, upper bound: 51.0413953
time: 99.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0640979, upper bound: 51.0410451
time: 104.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1710

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 554

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0434175, upper bound: 51.0453124
time: 109.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0434065, upper bound: 51.0453232
time: 93.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 205.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.72
Output dim: 1, lower bound: -51.0000946, upper bound: 51.0244026
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.72
Output dim: 1, lower bound: -51.0086291, upper bound: 51.0158515
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.72
Output dim: 1, lower bound: -51.0073332, upper bound: 51.0190804
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 205.72
Output dim: 1, lower bound: -51.0141388, upper bound: 51.0122490
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.72
Output dim: 1, lower bound: -51.0633247, upper bound: 51.0413953
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.72
Output dim: 1, lower bound: -51.0640979, upper bound: 51.0410451
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.72
Output dim: 1, lower bound: -51.0434175, upper bound: 51.0453124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.72
Output dim: 1, lower bound: -51.0434065, upper bound: 51.0453232

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 985

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 809

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9999373, upper bound: 51.0165790
time: 105.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9922603, upper bound: 51.0242384
time: 112.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -50.9931174, upper bound: 51.0004079
time: 122.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -50.9931174, upper bound: 51.0004079
time: 114.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1556

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -51.0068647, upper bound: 51.0050474
time: 109.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9932661, upper bound: 51.0185790
time: 91.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 519

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0611452, upper bound: 51.0390167
time: 125.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0609308, upper bound: 51.0392228
time: 114.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1570

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0637130, upper bound: 51.0403959
time: 110.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0636281, upper bound: 51.0406848
time: 90.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1566

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 773

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0383480, upper bound: 51.0451805
time: 110.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0432858, upper bound: 51.0402506
time: 97.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 546

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 534

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0370109, upper bound: 51.0404424
time: 95.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0385544, upper bound: 51.0388973
time: 99.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 197.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -50.9999373, upper bound: 51.0165790
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -50.9922603, upper bound: 51.0242384
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 197.43
Output dim: 1, lower bound: -50.9931174, upper bound: 51.0004079
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 197.43
Output dim: 1, lower bound: -50.9931174, upper bound: 51.0004079
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0068647, upper bound: 51.0050474
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -50.9932661, upper bound: 51.0185790
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0611452, upper bound: 51.0390167
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0609308, upper bound: 51.0392228
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0637130, upper bound: 51.0403959
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0636281, upper bound: 51.0406848
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0383480, upper bound: 51.0451805
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0432858, upper bound: 51.0402506
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0370109, upper bound: 51.0404424
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 197.43
Output dim: 1, lower bound: -51.0385544, upper bound: 51.0388973

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1727

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9995124, upper bound: 51.0165025
time: 781.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9998618, upper bound: 51.0161491
time: 137.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 522

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 624

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9893738, upper bound: 51.0213241
time: 100.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9893738, upper bound: 51.0213241
time: 95.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1695

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 779

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -50.9931268, upper bound: 51.0118907
time: 132.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9866212, upper bound: 51.0184400
time: 151.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1544

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0566570, upper bound: 51.0386308
time: 122.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0608204, upper bound: 51.0347114
time: 104.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 867

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 705

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0581823, upper bound: 51.0217111
time: 114.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0434098, upper bound: 51.0364919
time: 108.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0440224, upper bound: 51.0397499
time: 110.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0630655, upper bound: 51.0207455
time: 110.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 867

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1603

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0626606, upper bound: 51.0379541
time: 96.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0609078, upper bound: 51.0397078
time: 103.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1646

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0372301, upper bound: 51.0167496
time: 96.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0098913, upper bound: 51.0440611
time: 110.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 545

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1641

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0344254, upper bound: 51.0358491
time: 109.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0389792, upper bound: 51.0313144
time: 115.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 514

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0323347, upper bound: 51.0400313
time: 107.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0365916, upper bound: 51.0358863
time: 95.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0135113, upper bound: 51.0375767
time: 104.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -51.0372343, upper bound: 51.0138553
time: 91.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 198.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -50.9995124, upper bound: 51.0165025
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -50.9998618, upper bound: 51.0161491
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -50.9893738, upper bound: 51.0213241
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -50.9893738, upper bound: 51.0213241
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 198.11
Output dim: 1, lower bound: -50.9931268, upper bound: 51.0118907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -50.9866212, upper bound: 51.0184400
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0566570, upper bound: 51.0386308
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0608204, upper bound: 51.0347114
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0581823, upper bound: 51.0217111
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0434098, upper bound: 51.0364919
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0440224, upper bound: 51.0397499
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0630655, upper bound: 51.0207455
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0626606, upper bound: 51.0379541
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0609078, upper bound: 51.0397078
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0372301, upper bound: 51.0167496
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0098913, upper bound: 51.0440611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0344254, upper bound: 51.0358491
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0389792, upper bound: 51.0313144
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0323347, upper bound: 51.0400313
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0365916, upper bound: 51.0358863
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0135113, upper bound: 51.0375767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 198.11
Output dim: 1, lower bound: -51.0372343, upper bound: 51.0138553

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1668

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -50.9981019, upper bound: 51.0131418
time: 109.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9961595, upper bound: 51.0150920
time: 90.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1462

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -50.9998400, upper bound: 51.0114621
time: 114.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9951739, upper bound: 51.0161273
time: 119.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -68.9421158, 38.8874779, -68.9421158, 38.8874779, -107.8295898, 107.8295898
1: -36.4274559, 36.6292381, -36.4274559, 36.6292381, -73.0566940, 73.0566940
2: -32.1790504, 38.6885033, -32.1790504, 38.6885033, -70.8675537, 70.8675537
3: -35.7024994, 43.0713425, -35.7024994, 43.0713425, -78.7738342, 78.7738419
4: -41.7036819, 40.3332443, -41.7036819, 40.3332443, -82.0369263, 82.0369263
5: -37.2030602, 42.5177689, -37.2030602, 42.5177689, -79.7208252, 79.7208252
6: -62.4467316, 37.6486816, -62.4467316, 37.6486816, -100.0954132, 100.0954132
7: -44.4384727, 40.5514908, -44.4384727, 40.5514908, -84.9899597, 84.9899597
8: -50.0602112, 46.6771202, -50.0602112, 46.6771202, -96.7373276, 96.7373352
9: -40.9739571, 43.9675674, -40.9739571, 43.9675674, -84.9415283, 84.9415131
10: -63.3040619, 58.5418930, -63.3040619, 58.5418930, -121.8459473, 121.8459549
11: -59.4474640, 33.7324791, -59.4474640, 33.7324791, -93.1799469, 93.1799469
12: -60.8170738, 42.5191345, -60.8170738, 42.5191345, -103.3362045, 103.3362122
13: -65.5340881, 61.4166870, -65.5340881, 61.4166870, -126.9507599, 126.9507675
14: -99.5759964, 46.7554169, -99.5759964, 46.7554169, -146.3314209, 146.3314209
15: -47.9168205, 43.1619186, -47.9168205, 43.1619186, -91.0787354, 91.0787354
16: -62.7039909, 45.8929520, -62.7039909, 45.8929520, -108.5969391, 108.5969391
17: -96.4808960, 43.8861771, -96.4808960, 43.8861771, -140.3670654, 140.3670654
18: -59.4888763, 48.0116310, -59.4888763, 48.0116310, -107.5005035, 107.5005035
19: -48.6698952, 28.0165043, -48.6698952, 28.0165043, -76.6864014, 76.6864014
20: -46.6100960, 32.1936302, -46.6100960, 32.1936302, -78.8037262, 78.8037262
21: -58.3703270, 32.8232803, -58.3703270, 32.8232803, -91.1936035, 91.1936035
22: -61.1011162, 34.7029343, -61.1011162, 34.7029343, -95.8040466, 95.8040466
23: -47.2213783, 35.6981354, -47.2213783, 35.6981354, -82.9195099, 82.9195099
24: -57.4772491, 34.0789223, -57.4772491, 34.0789223, -91.5561676, 91.5561676
25: -51.6326752, 37.4441910, -51.6326752, 37.4441910, -89.0768661, 89.0768661
26: -70.1188583, 50.3679619, -70.1188583, 50.3679619, -120.4868164, 120.4868164
27: -57.0520706, 38.9679604, -57.0520706, 38.9679604, -96.0200348, 96.0200348
28: -47.7068176, 39.3720398, -47.7068176, 39.3720398, -87.0788574, 87.0788574
29: -60.2770233, 30.6403561, -60.2770233, 30.6403561, -90.9173737, 90.9173737
30: -58.5087509, 40.6055145, -58.5087509, 40.6055145, -99.1142578, 99.1142578
31: -59.5706291, 34.4693413, -59.5706291, 34.4693413, -94.0399628, 94.0399628
32: -61.1190567, 35.9123993, -61.1190567, 35.9123993, -97.0314560, 97.0314560
33: -86.5352478, 46.4728966, -86.5352478, 46.4728966, -133.0081482, 133.0081329
34: -75.2159729, 32.0503998, -75.2159729, 32.0503998, -107.2663727, 107.2663727
35: -71.0029526, 35.3009415, -71.0029526, 35.3009415, -106.3038940, 106.3038940
36: -71.9236603, 37.9698143, -71.9236603, 37.9698143, -109.8934631, 109.8934631
37: -102.2697372, 33.5482559, -102.2697372, 33.5482559, -135.8179932, 135.8179932
38: -87.0640488, 50.8727646, -87.0640488, 50.8727646, -137.9368134, 137.9368134
39: -97.9064407, 44.0797348, -97.9064407, 44.0797348, -141.9861450, 141.9861755
40: -78.5403137, 34.5549660, -78.5403137, 34.5549660, -113.0952759, 113.0952759
41: -64.6810303, 40.9101562, -64.6810303, 40.9101562, -105.5911865, 105.5911865
42: -48.5710068, 36.2564240, -48.5710068, 36.2564240, -84.8274231, 84.8274231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=564, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -50.9849368, upper bound: 50.9903787
time: 824.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -50.9849368, upper bound: 51.0189148
time: 147.13 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 974.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 974.40
Output dim: 1, lower bound: -50.9981019, upper bound: 51.0131418
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 974.40
Output dim: 1, lower bound: -50.9961595, upper bound: 51.0150920
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 974.40
Output dim: 1, lower bound: -50.9998400, upper bound: 51.0114621
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 974.40
Output dim: 1, lower bound: -50.9951739, upper bound: 51.0161273
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 974.40
Output dim: 1, lower bound: -50.9849368, upper bound: 50.9903787
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 974.40
Output dim: 1, lower bound: -50.9849368, upper bound: 51.0189148
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -50.9893738, upper bound: 51.0213241
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -50.9866212, upper bound: 51.0184400
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0566570, upper bound: 51.0386308
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0608204, upper bound: 51.0347114
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0581823, upper bound: 51.0217111
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0434098, upper bound: 51.0364919
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0440224, upper bound: 51.0397499
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0630655, upper bound: 51.0207455
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0626606, upper bound: 51.0379541
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0609078, upper bound: 51.0397078
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0372301, upper bound: 51.0167496
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0098913, upper bound: 51.0440611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0344254, upper bound: 51.0358491
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0389792, upper bound: 51.0313144
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0323347, upper bound: 51.0400313
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0365916, upper bound: 51.0358863
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0135113, upper bound: 51.0375767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 974.40
Output dim: 1, lower bound: -51.0372343, upper bound: 51.0138553

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 130.07 + 7472.66 = 7602.73 seconds
