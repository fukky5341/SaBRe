## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 3600 seconds
Split limit: 100
Threshold: 46.7201561769


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756)
1: (-27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899)
2: (-22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396)
3: (-25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715)
4: (-29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974)
5: (-28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624)
6: (-47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0804901, 61.0804787)
7: (-38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896)
8: (-35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060)
9: (-22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370)
10: (-41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275)
11: (-48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796)
12: (-43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567)
13: (-30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559)
14: (-77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135)
15: (-29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978)
16: (-46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752)
17: (-79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265)
18: (-45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209)
19: (-36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757)
20: (-31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987)
21: (-43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035)
22: (-37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158)
23: (-32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279)
24: (-30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335)
25: (-26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902)
26: (-50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358)
27: (-41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410)
28: (-32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099)
29: (-37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899)
30: (-41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926)
31: (-41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534)
32: (-36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222)
33: (-46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083)
34: (-44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718)
35: (-35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663)
36: (-39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803)
37: (-60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318)
38: (-56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467)
39: (-56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222)
40: (-58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.8190613, 74.8190613)
41: (-38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5981369, 57.5981369)
42: (-31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.82 + 67.82 = 70.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -46.7669231, upper bound: 46.7669231

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1360
type: B, layer: 1, pos: 1360
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1506
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1298
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1488
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 599

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7238566, upper bound: 46.7416727
time: 248.48 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7238566, upper bound: 46.7416727
time: 63.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 312.30 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 312.30
Output dim: 13, lower bound: -46.7238566, upper bound: 46.7416727
IS_B2, status: Status.UNKNOWN, split count: 1, time: 312.30
Output dim: 13, lower bound: -46.7238566, upper bound: 46.7416727

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -41.8456612, 32.1649475, -41.8294296, 32.1606140, -74.0062714, 73.9943771
1: -27.6181641, 29.8734379, -27.6106758, 29.8692970, -57.4874611, 57.4841156
2: -22.0847511, 25.5796204, -22.0783806, 25.5755768, -47.6603279, 47.6580009
3: -25.3986549, 31.8191681, -25.3857536, 31.8106747, -57.2093277, 57.2049217
4: -29.0450096, 28.7306824, -29.0404129, 28.7227764, -57.7677841, 57.7710953
5: -28.2739677, 29.0813847, -28.2680817, 29.0746174, -57.3485870, 57.3494644
6: -47.3105621, 14.1509933, -47.3015137, 14.1268616, -61.0305786, 61.0458488
7: -38.5162888, 25.7244720, -38.5035362, 25.7188568, -64.2351456, 64.2280121
8: -35.1641884, 29.4575996, -35.1607819, 29.4492874, -64.6134796, 64.6183777
9: -22.0117683, 30.4883804, -22.0055332, 30.4827480, -52.4945145, 52.4939117
10: -41.9406471, 32.8647385, -41.9346352, 32.8559380, -74.7965851, 74.7993774
11: -48.6073227, 19.9947166, -48.5826874, 19.9893398, -68.5966644, 68.5774078
12: -43.4435272, 25.0315475, -43.4347610, 25.0107841, -68.4543152, 68.4663086
13: -30.5039139, 38.7050743, -30.4962997, 38.6860504, -69.1899643, 69.2013702
14: -77.3831787, 5.2973309, -77.3514557, 5.2940359, -82.6772156, 82.6487885
15: -29.5099258, 36.1072617, -29.5025311, 36.0992050, -65.6091309, 65.6097946
16: -46.8024406, 28.1202736, -46.7880020, 28.1122093, -74.9146500, 74.9082794
17: -79.2430191, 16.1318798, -79.2014313, 16.1238632, -95.3668823, 95.3333130
18: -45.1071014, 18.2194748, -45.0978355, 18.2149067, -63.3220062, 63.3173103
19: -36.2812729, 11.3921471, -36.2571640, 11.3887062, -47.6699791, 47.6493111
20: -31.0021877, 15.6572495, -30.9919834, 15.6514816, -46.6536713, 46.6492310
21: -43.6056633, 14.5188847, -43.5814743, 14.5152254, -58.1208878, 58.1003571
22: -37.6534615, 21.3252449, -37.6265717, 21.3212910, -58.9747543, 58.9518166
23: -32.3478470, 18.2297497, -32.3255768, 18.2269268, -50.5747757, 50.5553284
24: -30.8343163, 16.2665977, -30.8035316, 16.2632179, -47.0975342, 47.0701294
25: -26.9283638, 24.1839905, -26.9072361, 24.1789513, -51.1073151, 51.0912247
26: -50.0236969, 24.3606243, -50.0053978, 24.3525009, -74.3761978, 74.3660202
27: -41.6404572, 12.1827936, -41.6199799, 12.1794376, -53.8198929, 53.8027725
28: -32.7270355, 22.9256363, -32.6992836, 22.9207535, -55.6477890, 55.6249199
29: -37.8097153, 17.8009167, -37.7759247, 17.7970562, -55.6067734, 55.5768433
30: -41.0743713, 21.7360802, -41.0497589, 21.7309303, -62.8053017, 62.7858391
31: -41.7926483, 14.3585730, -41.7607498, 14.3546886, -56.1473389, 56.1193237
32: -36.9438438, 18.5715485, -36.9356537, 18.5536499, -55.4974937, 55.5072021
33: -46.8173752, 30.6395683, -46.8076591, 30.6317711, -77.4491425, 77.4472275
34: -44.5166550, 25.8719139, -44.5083160, 25.8653488, -70.3820038, 70.3802338
35: -35.8743134, 29.5228519, -35.8644524, 29.5196018, -65.3939133, 65.3873062
36: -39.7069969, 26.7940369, -39.7000275, 26.7801399, -66.4871368, 66.4940643
37: -60.5423470, 23.0086746, -60.5281258, 22.9806976, -83.5230408, 83.5368042
38: -56.7407799, 26.3229198, -56.7321663, 26.2961464, -83.0369263, 83.0550842
39: -56.9080963, 21.4076443, -56.8972206, 21.3922405, -78.3003387, 78.3048630
40: -58.8010139, 16.5350437, -58.7931099, 16.5029240, -74.7559586, 74.7808685
41: -38.5501785, 19.0254555, -38.5407677, 19.0043850, -57.5545654, 57.5662231
42: -31.9946003, 16.9058418, -31.9893723, 16.8922749, -48.8868752, 48.8952141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1460
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 1360
type: A, layer: 1, pos: 1360
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1532
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1298
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1488
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 601

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6842690, upper bound: 46.7195519
time: 59.61 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6842690, upper bound: 46.7195519
time: 48.56 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -41.8393059, 32.1638985, -41.8840179, 32.2059021, -74.0452118, 74.0479126
1: -27.6127434, 29.8723984, -27.6302872, 29.9255123, -57.5382538, 57.5026855
2: -22.0800476, 25.5791950, -22.1010551, 25.6337471, -47.7137947, 47.6802521
3: -25.3906326, 31.8191204, -25.4137192, 31.8801346, -57.2707672, 57.2328415
4: -29.0436268, 28.7318439, -29.0666733, 28.7805748, -57.8242035, 57.7985153
5: -28.2717361, 29.0810585, -28.2911396, 29.1216583, -57.3933945, 57.3722000
6: -47.3125687, 14.1592064, -47.4597359, 14.1820765, -61.0853195, 61.2131195
7: -38.5016937, 25.7211971, -38.5019226, 25.7814960, -64.2831879, 64.2231216
8: -35.1537209, 29.4561443, -35.1689301, 29.5209427, -64.6746674, 64.6250763
9: -22.0121975, 30.4880142, -22.1061363, 30.5109863, -52.5231857, 52.5941505
10: -41.9379082, 32.8623276, -42.0092812, 32.9081001, -74.8460083, 74.8716125
11: -48.5966034, 19.9906769, -48.6215096, 20.1455574, -68.7421570, 68.6121826
12: -43.4420700, 25.0132866, -43.6057281, 25.0311317, -68.4732056, 68.6190186
13: -30.5007648, 38.7026520, -30.7237663, 38.7216377, -69.2224045, 69.4264221
14: -77.3869324, 5.2962770, -77.4315491, 5.4166660, -82.8035965, 82.7278290
15: -29.5048733, 36.0937157, -29.5279446, 36.1082649, -65.6131363, 65.6216583
16: -46.7963066, 28.1166000, -46.8629036, 28.2041817, -75.0004883, 74.9795074
17: -79.2558289, 16.1268539, -79.3055725, 16.3627014, -95.6185303, 95.4324265
18: -45.1023979, 18.2181873, -45.1358490, 18.3325386, -63.4349365, 63.3540344
19: -36.2913132, 11.3897371, -36.3199120, 11.5095997, -47.8009109, 47.7096481
20: -31.0011997, 15.6560268, -31.0249271, 15.7385769, -46.7397766, 46.6809540
21: -43.6046600, 14.5156393, -43.6386337, 14.6597452, -58.2644043, 58.1542740
22: -37.6594658, 21.3233566, -37.6996994, 21.4392300, -59.0986938, 59.0230560
23: -32.3509941, 18.2277660, -32.3786011, 18.3763790, -50.7273712, 50.6063690
24: -30.8451633, 16.2662621, -30.8835030, 16.4512348, -47.2963982, 47.1497650
25: -26.9344368, 24.1829090, -26.9696712, 24.3386383, -51.2730751, 51.1525803
26: -50.0260239, 24.3587780, -50.0751686, 24.4488907, -74.4749146, 74.4339447
27: -41.6464882, 12.1809855, -41.6739120, 12.3392086, -53.9856949, 53.8548965
28: -32.7386971, 22.9241123, -32.7654076, 23.0830841, -55.8217812, 55.6895218
29: -37.8194008, 17.7964420, -37.8632545, 17.9456768, -55.7650757, 55.6596985
30: -41.0719528, 21.7330704, -41.0974617, 21.9197292, -62.9916840, 62.8305321
31: -41.8054657, 14.3560858, -41.8374100, 14.5000076, -56.3054733, 56.1934967
32: -36.9432983, 18.5774803, -37.1473389, 18.6164627, -55.5597610, 55.7248192
33: -46.8166504, 30.6370754, -46.9938660, 30.6573734, -77.4740219, 77.6309433
34: -44.5145416, 25.8624096, -44.6129913, 25.8810978, -70.3956375, 70.4754028
35: -35.8731003, 29.5205116, -35.9789200, 29.5292320, -65.4023285, 65.4994354
36: -39.7042084, 26.7873192, -39.8627548, 26.7939453, -66.4981537, 66.6500702
37: -60.5404930, 23.0082626, -60.7556076, 23.0179787, -83.5584717, 83.7638702
38: -56.7338867, 26.3123837, -56.9302559, 26.3307076, -83.0645905, 83.2426376
39: -56.9058609, 21.3999481, -57.1582832, 21.4103928, -78.3162537, 78.5582275
40: -58.7968330, 16.5476761, -59.0189209, 16.5648727, -74.8126984, 75.0193939
41: -38.5515022, 19.0325661, -38.7291565, 19.0622406, -57.6137428, 57.7617226
42: -31.9960461, 16.9067822, -32.1136169, 16.9361973, -48.9322433, 49.0204010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=361, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1468
type: A, layer: 1, pos: 1468
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1460
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1360
type: B, layer: 1, pos: 1360
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1383
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1505
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1505
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1532
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1506
type: A, layer: 1, pos: 1517
type: B, layer: 1, pos: 1517
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1488
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 601

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6842690, upper bound: 46.7195519
time: 102.04 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.7195516, upper bound: 46.7195519
time: 70.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 175.14 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 175.14
Output dim: 13, lower bound: -46.6842690, upper bound: 46.7195519
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 175.14
Output dim: 13, lower bound: -46.6842690, upper bound: 46.7195519
IS_B2_B1, status: Status.VERIFIED, split count: 2, time: 175.14
Output dim: 13, lower bound: -46.6842690, upper bound: 46.7195519
IS_B2_B2, status: Status.VERIFIED, split count: 2, time: 175.14
Output dim: 13, lower bound: -46.7195516, upper bound: 46.7195519

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 70.64 + 597.86 = 668.50 seconds
