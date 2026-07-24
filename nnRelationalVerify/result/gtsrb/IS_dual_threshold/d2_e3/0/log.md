## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 3600 seconds
Split limit: 100
Threshold: 14.8088132631


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4408264, 54.4408264)
1: (-7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5702744, 36.5702744)
2: (-4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2446671, 33.2446671)
3: (-8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5336609, 32.5336609)
4: (-10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4993439, 43.4993515)
5: (-11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2652054, 38.2652054)
6: (-38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5373840, 44.5373840)
7: (-15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8042145, 41.8042145)
8: (-15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1943817, 47.1943817)
9: (-10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8660660, 35.8660660)
10: (-28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6891479, 50.6891479)
11: (-35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5255661, 49.5255661)
12: (-49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2132187, 44.2132187)
13: (-28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6185379, 49.6185455)
14: (-71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952)
15: (-17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268)
16: (-27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5910492, 48.5910416)
17: (-71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727)
18: (-34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6379433, 40.6379395)
19: (-25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9775314, 29.9775314)
20: (-26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1457596, 29.1457558)
21: (-31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4689636, 40.4689713)
22: (-33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5501099, 38.5501022)
23: (-26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2848053, 35.2848015)
24: (-23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8603859, 32.8603897)
25: (-29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4305496, 34.4305496)
26: (-42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8384476, 43.8384476)
27: (-26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8176346, 37.8176308)
28: (-29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6648636, 36.6648712)
29: (-32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889)
30: (-37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515)
31: (-31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6741943, 37.6741982)
32: (-33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604)
33: (-44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2882080, 57.2882156)
34: (-50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.4060440, 42.4060516)
35: (-41.0334129, 7.0879445, -41.0334129, 7.0879445, -44.0117950, 44.0118027)
36: (-44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6235657, 45.6235580)
37: (-59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2817917, 55.2817993)
38: (-50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959)
39: (-52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457)
40: (-47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2302246, 53.2302170)
41: (-31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5972900, 45.5972900)
42: (-27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8905182, 36.8905182)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.80 + 46.95 = 49.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -14.8236368, upper bound: 14.8236369

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7831672, upper bound: 14.8227297
time: 41.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8227295, upper bound: 14.8227297
time: 39.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 81.49 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 81.49
Output dim: 2, lower bound: -14.7831672, upper bound: 14.8227297
IS_A2, status: Status.UNKNOWN, split count: 1, time: 81.49
Output dim: 2, lower bound: -14.8227295, upper bound: 14.8227297

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -23.8200340, 32.8104019, -23.9139404, 32.8886909, -54.2744293, 54.2854385
1: -7.6348286, 32.3249931, -7.6826839, 32.3447876, -36.4805145, 36.5140381
2: -4.6703672, 31.7237148, -4.7444954, 31.8038216, -33.1070251, 33.0978622
3: -8.8604088, 28.8420410, -8.9329777, 28.9137573, -32.4036407, 32.4135780
4: -10.0149632, 35.0119896, -10.1001940, 35.0861435, -43.3413773, 43.3573837
5: -11.0327234, 29.8299065, -11.1242447, 29.9296360, -38.0805130, 38.0717316
6: -38.6651726, 7.4404716, -38.7306976, 7.4824791, -44.3368912, 44.4137878
7: -15.2236671, 30.6141224, -15.2948980, 30.6382732, -41.6844177, 41.7263336
8: -15.3596554, 34.4797211, -15.4439659, 34.5795097, -47.0254974, 47.0115204
9: -10.2720032, 27.0734272, -10.3563147, 27.1589603, -35.7024231, 35.7085648
10: -28.2418785, 23.7229958, -28.4638805, 23.8916321, -50.2992249, 50.3543625
11: -35.5341492, 14.1169739, -35.7824516, 14.2441463, -49.1305847, 49.2445068
12: -49.2253418, 1.7568040, -49.4161606, 1.9388647, -43.8456955, 43.8577118
13: -28.7960529, 21.2200565, -28.8252850, 21.3124657, -49.5160446, 49.4589996
14: -70.8625565, -6.5270519, -71.0062408, -6.4043999, -64.4581604, 64.4791870
15: -17.3306160, 24.5765648, -17.4043999, 24.6822968, -42.0129128, 41.9809647
16: -27.3658562, 23.5482597, -27.5251694, 23.6351490, -48.3544540, 48.3914948
17: -71.1021423, -3.9656200, -71.1651917, -3.8437538, -67.2583923, 67.1995697
18: -34.6756592, 11.6766415, -34.7467308, 11.6908531, -40.5061989, 40.5606461
19: -25.6255798, 5.2320070, -25.7025719, 5.2637968, -29.8672867, 29.8939285
20: -26.3765182, 4.2942433, -26.4712219, 4.3496590, -29.0116501, 29.0337334
21: -31.2173977, 10.0506058, -31.3633976, 10.1229992, -40.2185287, 40.3059845
22: -33.5709229, 6.9354715, -33.6187668, 6.9814410, -38.4430084, 38.4487610
23: -26.8242912, 8.8434830, -26.9136868, 8.8854675, -35.2024689, 35.2138672
24: -23.1832848, 9.8597240, -23.2464256, 9.8757219, -32.7484131, 32.7736282
25: -29.1620903, 6.0607905, -29.2138672, 6.1068034, -34.3572464, 34.3300285
26: -42.8402481, 7.6997819, -42.9492455, 7.7988319, -43.6057053, 43.6255417
27: -26.5723686, 11.4790955, -26.6384869, 11.4901829, -37.6965714, 37.7511406
28: -29.5724773, 7.1479292, -29.6037407, 7.1747084, -36.6014862, 36.5865173
29: -32.5681610, 8.9574785, -32.6097069, 9.0051003, -41.5732613, 41.5671844
30: -37.5130920, 6.8974552, -37.5894165, 6.9618387, -44.4749298, 44.4868698
31: -31.3045006, 7.2679019, -31.4078979, 7.2910399, -37.5769272, 37.5824852
32: -33.6132698, 6.6683779, -33.7162781, 6.7370949, -40.3503647, 40.3846550
33: -43.8425407, 15.9221821, -43.9448395, 16.0211964, -57.0814362, 57.0873947
34: -50.6630402, -4.3098717, -50.7234993, -4.2418613, -42.2870865, 42.2867126
35: -40.8822060, 6.9874539, -40.9630165, 7.0779605, -43.8457794, 43.8425598
36: -44.4456520, 5.3997064, -44.4814072, 5.4405584, -45.5363922, 45.5341797
37: -59.3944473, 2.3496342, -59.4563293, 2.3835797, -55.1371918, 55.1568298
38: -50.8619690, 8.5700083, -50.9236374, 8.6224356, -59.4844055, 59.4936447
39: -52.1131248, 14.8367434, -52.1733017, 14.8980703, -67.0111923, 67.0100479
40: -47.8057022, 8.2956200, -47.8785553, 8.3272676, -53.1057663, 53.1467438
41: -31.7983932, 15.2275238, -31.8898392, 15.2608519, -45.4404449, 45.5063782
42: -27.0821419, 10.0613213, -27.1948986, 10.1473207, -36.6196518, 36.6799507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=179, inp2_unstable=180, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7434115, upper bound: 14.8208456
time: 34.06 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7434115, upper bound: 14.8208456
time: 39.51 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -23.9600182, 32.9037514, -23.9643059, 32.9053726, -54.4005280, 54.4321899
1: -7.7053843, 32.3460426, -7.7079821, 32.3466187, -36.5657043, 36.5494308
2: -4.7966986, 31.8051891, -4.7997265, 31.8061619, -33.1780243, 33.2406120
3: -8.9756851, 28.9225121, -8.9827805, 28.9240742, -32.4648132, 32.5193977
4: -10.1546116, 35.0906792, -10.1615257, 35.0924492, -43.4485016, 43.4795380
5: -11.1897974, 29.9429245, -11.1934032, 29.9449673, -38.1852112, 38.2545013
6: -38.7545967, 7.4831352, -38.7578888, 7.4978313, -44.5398178, 44.5105820
7: -15.3257589, 30.6443100, -15.3300610, 30.6460705, -41.7738190, 41.7931442
8: -15.5035944, 34.5881805, -15.5090904, 34.5898552, -47.1243134, 47.1855621
9: -10.3590641, 27.2182198, -10.3605080, 27.2234745, -35.8587723, 35.7706604
10: -28.4720478, 24.0221367, -28.4735451, 24.0337715, -50.6739197, 50.4939270
11: -35.8061066, 14.3472614, -35.8091812, 14.3566418, -49.5094604, 49.4332657
12: -49.4244461, 2.0734153, -49.4269180, 2.0856233, -44.1960144, 44.1177673
13: -28.8301258, 21.3571091, -28.8325081, 21.3617592, -49.6096039, 49.6059799
14: -71.0138855, -6.3120136, -71.0165100, -6.3058243, -64.7080612, 64.7044983
15: -17.4423561, 24.7014122, -17.4513683, 24.7036343, -42.1459885, 42.1527786
16: -27.5410385, 23.6989746, -27.5444431, 23.7085304, -48.5726547, 48.4786453
17: -71.1750641, -3.7562943, -71.1774902, -3.7475204, -67.4275436, 67.4211960
18: -34.7622337, 11.6959476, -34.7643166, 11.6971512, -40.6133347, 40.6206284
19: -25.7187595, 5.2889185, -25.7207680, 5.2918706, -29.9459000, 29.9616623
20: -26.4857273, 4.3893862, -26.4873161, 4.3933816, -29.1360855, 29.0831985
21: -31.3792229, 10.1814308, -31.3817215, 10.1864414, -40.4626007, 40.3545837
22: -33.6279678, 7.0045104, -33.6375885, 7.0075049, -38.5256271, 38.5322037
23: -26.9233627, 8.9132233, -26.9250565, 8.9159203, -35.2668610, 35.2677689
24: -23.2745075, 9.8779430, -23.2792168, 9.8790760, -32.8427658, 32.8667221
25: -29.2264709, 6.1348910, -29.2294350, 6.1376472, -34.4075394, 34.4335861
26: -42.9674950, 7.8641133, -42.9709282, 7.8706570, -43.8239899, 43.7674942
27: -26.6652889, 11.4925690, -26.6726189, 11.4937897, -37.8045425, 37.8090553
28: -29.6159401, 7.1807256, -29.6176529, 7.1832609, -36.6461411, 36.6716614
29: -32.6178398, 9.0350246, -32.6194534, 9.0390930, -41.6569328, 41.6544800
30: -37.6043549, 6.9989080, -37.6069107, 7.0037870, -44.6081429, 44.6058197
31: -31.4270287, 7.3072996, -31.4291191, 7.3094487, -37.6309090, 37.7229462
32: -33.7274666, 6.7820215, -33.7293320, 6.7875910, -40.5150566, 40.5113525
33: -44.0149040, 16.0346527, -44.0225067, 16.0362759, -57.2166443, 57.2769470
34: -50.7640839, -4.2363529, -50.7685280, -4.2353039, -42.3589096, 42.3987808
35: -41.0208969, 7.0862169, -41.0270691, 7.0870662, -43.9461975, 44.0032501
36: -44.4974442, 5.4519620, -44.5018654, 5.4547906, -45.6142883, 45.6123123
37: -59.4826851, 2.3964610, -59.4869232, 2.3978548, -55.2558823, 55.2716370
38: -50.9539223, 8.6419563, -50.9587860, 8.6470385, -59.6009598, 59.6007423
39: -52.1983833, 14.9233465, -52.2031097, 14.9265823, -67.1249695, 67.1264572
40: -47.9129562, 8.3327303, -47.9167252, 8.3353148, -53.2196808, 53.2209854
41: -31.9154892, 15.2806940, -31.9183769, 15.2831659, -45.5888138, 45.5855789
42: -27.2081490, 10.2072611, -27.2107315, 10.2150040, -36.8819962, 36.8696289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=179, inp2_unstable=180, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
time: 19.50 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455
time: 39.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 61.07 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 61.07
Output dim: 2, lower bound: -14.7434115, upper bound: 14.8208456
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 61.07
Output dim: 2, lower bound: -14.7434115, upper bound: 14.8208456
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 61.07
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 61.07
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -23.6473656, 32.7435646, -23.8271236, 32.8826599, -54.0970612, 54.1347427
1: -7.5463972, 32.2867012, -7.6392317, 32.3389893, -36.3853073, 36.4318085
2: -4.5523834, 31.6734161, -4.6856680, 31.7984619, -32.9833679, 32.9878387
3: -8.7778187, 28.7919159, -8.8933811, 28.9060783, -32.3134613, 32.3236847
4: -9.8837128, 34.9512138, -10.0368156, 35.0799179, -43.2003860, 43.2314453
5: -10.9465694, 29.7675247, -11.0824165, 29.9195213, -37.9847107, 37.9670486
6: -38.6150703, 7.2876434, -38.7217102, 7.4072151, -44.2057800, 44.2482376
7: -15.1209784, 30.5689735, -15.2442904, 30.6339893, -41.5750122, 41.6289749
8: -15.2004375, 34.3986053, -15.3672905, 34.5697021, -46.8556671, 46.8537903
9: -10.1822748, 27.0348034, -10.3133259, 27.1538620, -35.5961151, 35.6112061
10: -28.1595612, 23.6746655, -28.4237671, 23.8788605, -50.1865158, 50.2329254
11: -35.4724083, 14.0214958, -35.7691879, 14.1994934, -49.0076370, 49.1248627
12: -49.1694489, 1.6340556, -49.4081421, 1.8826442, -43.7291183, 43.7201996
13: -28.7104568, 21.1631374, -28.7864685, 21.2969952, -49.4062347, 49.3470230
14: -70.6940155, -6.5902271, -70.9243088, -6.4164295, -64.2775879, 64.3340836
15: -17.2449932, 24.5401726, -17.3643131, 24.6728077, -41.9178009, 41.9044876
16: -27.2762642, 23.5166016, -27.4835739, 23.6277790, -48.2435455, 48.3053284
17: -71.0493927, -3.9978943, -71.1427917, -3.8531666, -67.1962280, 67.1448975
18: -34.6340561, 11.6163597, -34.7344971, 11.6619205, -40.4280663, 40.4820290
19: -25.5743904, 5.1805944, -25.6888866, 5.2378454, -29.7857208, 29.8243446
20: -26.3349819, 4.2229042, -26.4631405, 4.3146563, -28.9355164, 28.9537582
21: -31.1537724, 9.9546366, -31.3479290, 10.0753202, -40.1031647, 40.1933594
22: -33.5054245, 6.8344631, -33.6060333, 6.9309411, -38.3262024, 38.3353271
23: -26.7728043, 8.7801399, -26.9014778, 8.8546047, -35.1148682, 35.1338844
24: -23.1293964, 9.8096676, -23.2346420, 9.8512497, -32.6661453, 32.7118607
25: -29.1070595, 5.9768634, -29.2028122, 6.0660615, -34.2602234, 34.2348213
26: -42.7682648, 7.5988579, -42.9371452, 7.7489214, -43.4782104, 43.5069351
27: -26.5147152, 11.4066992, -26.6268291, 11.4550323, -37.5964432, 37.6646118
28: -29.5127831, 7.0633817, -29.5927238, 7.1328874, -36.4999008, 36.4903793
29: -32.5073395, 8.8433275, -32.5988655, 8.9497709, -41.4571114, 41.4421921
30: -37.4493256, 6.7736712, -37.5799446, 6.9009352, -44.3502617, 44.3536148
31: -31.2537994, 7.2320437, -31.3939571, 7.2742987, -37.4974289, 37.5243797
32: -33.5652466, 6.5754738, -33.7030029, 6.6939850, -40.2592316, 40.2784767
33: -43.7620316, 15.8692112, -43.9189301, 15.9960556, -56.9645996, 56.9914932
34: -50.6047745, -4.3974462, -50.7084579, -4.2852678, -42.1782074, 42.1729126
35: -40.8200989, 6.9264984, -40.9479904, 7.0478096, -43.7476654, 43.7570267
36: -44.3934555, 5.3087716, -44.4714890, 5.3942823, -45.4346466, 45.4295731
37: -59.3146629, 2.3030686, -59.4337082, 2.3624053, -55.0302353, 55.0816498
38: -50.8077812, 8.5114269, -50.9108810, 8.5950413, -59.4028244, 59.4223099
39: -52.0314026, 14.8193970, -52.1449051, 14.8895912, -66.9209900, 66.9643021
40: -47.7561913, 8.2587500, -47.8581772, 8.3150177, -53.0360107, 53.0746231
41: -31.7409058, 15.1300020, -31.8748398, 15.2131567, -45.3312378, 45.3896179
42: -27.0405540, 9.9699821, -27.1838951, 10.1040211, -36.5315857, 36.5745049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=180, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7391297, upper bound: 14.7711753
time: 39.24 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7391297, upper bound: 14.8189533
time: 38.57 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -23.8144550, 32.8096428, -23.9115753, 32.8883705, -54.2437439, 54.2826996
1: -7.6310825, 32.3242760, -7.6811810, 32.3445129, -36.4524841, 36.5109444
2: -4.6668220, 31.7230244, -4.7429461, 31.8035316, -33.0833817, 33.0954361
3: -8.8569221, 28.8414001, -8.9315681, 28.9135151, -32.3840866, 32.4115105
4: -10.0098114, 35.0112686, -10.0980968, 35.0858612, -43.3261337, 43.3543167
5: -11.0297270, 29.8290901, -11.1228733, 29.9292717, -38.0636597, 38.0695496
6: -38.6636276, 7.4359894, -38.7300758, 7.4802933, -44.3328094, 44.3989792
7: -15.2203808, 30.6135502, -15.2935553, 30.6380291, -41.6521301, 41.7241592
8: -15.3539219, 34.4778824, -15.4414568, 34.5787468, -46.9968185, 47.0073395
9: -10.2665129, 27.0728436, -10.3540049, 27.1587105, -35.6955185, 35.7130814
10: -28.2331657, 23.7214928, -28.4604111, 23.8910522, -50.2837601, 50.3734055
11: -35.5319061, 14.1143093, -35.7815132, 14.2428188, -49.1385498, 49.2375183
12: -49.2235107, 1.7522359, -49.4153633, 1.9370337, -43.8415680, 43.8195572
13: -28.7931347, 21.2180920, -28.8240700, 21.3116684, -49.5096512, 49.4701462
14: -70.8561783, -6.5284901, -71.0033035, -6.4049664, -64.4512100, 64.4748154
15: -17.3249321, 24.5755196, -17.4019394, 24.6819077, -42.0068398, 41.9774590
16: -27.3582497, 23.5471764, -27.5219479, 23.6346912, -48.3330536, 48.3871231
17: -71.0985031, -3.9663830, -71.1636887, -3.8441029, -67.2544022, 67.1973038
18: -34.6736832, 11.6739349, -34.7459030, 11.6896667, -40.5050583, 40.5567284
19: -25.6239471, 5.2295089, -25.7019005, 5.2626619, -29.8684998, 29.8899307
20: -26.3755569, 4.2906933, -26.4708061, 4.3481183, -29.0087280, 29.0181427
21: -31.2153034, 10.0461464, -31.3625641, 10.1211529, -40.2231827, 40.2989883
22: -33.5690002, 6.9322615, -33.6179504, 6.9800997, -38.4397354, 38.4192505
23: -26.8227234, 8.8403635, -26.9130383, 8.8842058, -35.2042694, 35.2089157
24: -23.1818848, 9.8576889, -23.2458515, 9.8748188, -32.7509689, 32.7695312
25: -29.1605663, 6.0575085, -29.2132359, 6.1054540, -34.3539200, 34.3240814
26: -42.8386154, 7.6966071, -42.9485474, 7.7973371, -43.6030655, 43.5915909
27: -26.5712967, 11.4770317, -26.6380386, 11.4893370, -37.7027664, 37.7462387
28: -29.5712471, 7.1442575, -29.6032562, 7.1732373, -36.5989990, 36.5822067
29: -32.5665359, 8.9543276, -32.6090164, 9.0035343, -41.5700684, 41.5633430
30: -37.5117569, 6.8927450, -37.5888519, 6.9594498, -44.4712067, 44.4815979
31: -31.3026218, 7.2652726, -31.4071007, 7.2899489, -37.5823326, 37.5758667
32: -33.6115685, 6.6644497, -33.7155685, 6.7354841, -40.3470535, 40.3800201
33: -43.8399239, 15.9164524, -43.9437332, 16.0188580, -57.0756073, 57.0719070
34: -50.6614456, -4.3144989, -50.7228394, -4.2440958, -42.2833328, 42.2538071
35: -40.8806343, 6.9831529, -40.9623604, 7.0761352, -43.8416901, 43.8290634
36: -44.4446335, 5.3953104, -44.4809799, 5.4384041, -45.5330048, 45.5090103
37: -59.3922577, 2.3454037, -59.4553909, 2.3818750, -55.1329193, 55.1349792
38: -50.8604469, 8.5637236, -50.9230156, 8.6198568, -59.4803047, 59.4867401
39: -52.1107712, 14.8354464, -52.1722336, 14.8975573, -67.0083313, 67.0076828
40: -47.8025055, 8.2938633, -47.8772430, 8.3265877, -53.0970383, 53.1549377
41: -31.7966194, 15.2247143, -31.8891258, 15.2595634, -45.4373322, 45.4946747
42: -27.0803013, 10.0588818, -27.1941662, 10.1462641, -36.6165771, 36.6716614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=180, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7391297, upper bound: 14.7711753
time: 40.36 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7769932, upper bound: 14.8189533
time: 37.20 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -23.9576893, 32.9034805, -23.9586868, 32.9046326, -54.3977356, 54.4015121
1: -7.7038975, 32.3457489, -7.7042494, 32.3459015, -36.5625763, 36.5213470
2: -4.7951279, 31.8049297, -4.7961602, 31.8054066, -33.1756363, 33.2169724
3: -8.9742699, 28.9222107, -8.9792747, 28.9234314, -32.4627533, 32.4998703
4: -10.1525345, 35.0903816, -10.1563892, 35.0917702, -43.4453735, 43.4642715
5: -11.1884556, 29.9425755, -11.1903973, 29.9441147, -38.1829987, 38.2376175
6: -38.7539787, 7.4809275, -38.7563133, 7.4934044, -44.5250397, 44.5064392
7: -15.3244171, 30.6440811, -15.3267670, 30.6455021, -41.7715759, 41.7608414
8: -15.5010643, 34.5874405, -15.5033703, 34.5880203, -47.1201477, 47.1568451
9: -10.3567629, 27.2179699, -10.3550167, 27.2228928, -35.8633347, 35.7637863
10: -28.4685440, 24.0215263, -28.4648170, 24.0322571, -50.6929321, 50.4784927
11: -35.8051643, 14.3459187, -35.8069687, 14.3540058, -49.5024872, 49.4411926
12: -49.4237213, 2.0715489, -49.4250679, 2.0810914, -44.1578369, 44.1136932
13: -28.8289299, 21.3563290, -28.8296089, 21.3598347, -49.6208115, 49.5996170
14: -71.0109100, -6.3126411, -71.0101013, -6.3072319, -64.7036743, 64.6974640
15: -17.4398766, 24.7009621, -17.4457016, 24.7026081, -42.1424866, 42.1466637
16: -27.5377979, 23.6985207, -27.5367889, 23.7073936, -48.5683136, 48.4571838
17: -71.1735458, -3.7566452, -71.1738205, -3.7483063, -67.4252396, 67.4171753
18: -34.7614212, 11.6947536, -34.7623596, 11.6944370, -40.6094437, 40.6195030
19: -25.7180653, 5.2877426, -25.7191277, 5.2893639, -29.9419403, 29.9628754
20: -26.4853115, 4.3878317, -26.4863625, 4.3898358, -29.1204681, 29.0802879
21: -31.3783417, 10.1795549, -31.3796082, 10.1819954, -40.4555893, 40.3592072
22: -33.6271744, 7.0032053, -33.6356621, 7.0042839, -38.4960709, 38.5289230
23: -26.9226856, 8.9119625, -26.9234715, 8.9128304, -35.2619095, 35.2695847
24: -23.2739277, 9.8770418, -23.2778282, 9.8770304, -32.8386421, 32.8693123
25: -29.2258568, 6.1335626, -29.2279167, 6.1343784, -34.4015961, 34.4302063
26: -42.9668770, 7.8625560, -42.9692993, 7.8674421, -43.7900391, 43.7648468
27: -26.6648903, 11.4917374, -26.6715260, 11.4917431, -37.7996674, 37.8152618
28: -29.6154404, 7.1792526, -29.6164417, 7.1795731, -36.6417770, 36.6691666
29: -32.6171646, 9.0334167, -32.6178169, 9.0359478, -41.6531143, 41.6512337
30: -37.6037750, 6.9964952, -37.6055336, 6.9990978, -44.6028748, 44.6020279
31: -31.4262428, 7.3062000, -31.4272461, 7.3068261, -37.6242905, 37.7283707
32: -33.7267761, 6.7803774, -33.7276344, 6.7836189, -40.5103951, 40.5080109
33: -44.0137863, 16.0323601, -44.0198975, 16.0304928, -57.2011108, 57.2710495
34: -50.7634315, -4.2385855, -50.7669220, -4.2399402, -42.3260117, 42.3949738
35: -41.0202408, 7.0843239, -41.0255241, 7.0827870, -43.9325714, 43.9991379
36: -44.4969749, 5.4497776, -44.5007858, 5.4504213, -45.5891266, 45.6089325
37: -59.4817200, 2.3947096, -59.4847336, 2.3936315, -55.2340698, 55.2674026
38: -50.9533043, 8.6393633, -50.9572601, 8.6407909, -59.5940933, 59.5966225
39: -52.1973495, 14.9227991, -52.2007370, 14.9252911, -67.1226425, 67.1235352
40: -47.9115982, 8.3320656, -47.9135590, 8.3335800, -53.2278748, 53.2122955
41: -31.9147911, 15.2793875, -31.9166851, 15.2803526, -45.5771103, 45.5824127
42: -27.2074280, 10.2061615, -27.2088985, 10.2125692, -36.8736877, 36.8665581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=179, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
time: 40.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8189531, upper bound: 14.8189535
time: 77.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 120.66 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 120.66
Output dim: 2, lower bound: -14.7391297, upper bound: 14.7711753
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 120.66
Output dim: 2, lower bound: -14.7391297, upper bound: 14.8189533
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 120.66
Output dim: 2, lower bound: -14.7391297, upper bound: 14.7711753
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 120.66
Output dim: 2, lower bound: -14.7769932, upper bound: 14.8189533
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 120.66
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 120.66
Output dim: 2, lower bound: -14.8189531, upper bound: 14.8189535

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -23.6389561, 32.7423515, -23.8630676, 33.0269585, -54.2415771, 54.1708450
1: -7.5404892, 32.2859802, -7.6488676, 32.3973198, -36.4392014, 36.4491768
2: -4.5463743, 31.6724110, -4.7019463, 31.9291191, -33.1163254, 33.0019493
3: -8.7718811, 28.7901917, -8.9078665, 29.0798569, -32.4785004, 32.3309746
4: -9.8788595, 34.9498100, -10.0476284, 35.1840401, -43.3044510, 43.2440414
5: -10.9392433, 29.7656822, -11.0991173, 30.1051407, -38.1644897, 37.9784698
6: -38.6117172, 7.2825804, -38.7623787, 7.4183540, -44.2082214, 44.3249969
7: -15.1130085, 30.5666122, -15.2724657, 30.7300167, -41.6703644, 41.6511002
8: -15.1923485, 34.3971786, -15.3704329, 34.6862831, -46.9788208, 46.8601913
9: -10.1801405, 27.0280457, -10.3709993, 27.1677418, -35.6162720, 35.6586952
10: -28.1582375, 23.6650124, -28.6326828, 23.8908024, -50.1955261, 50.4374847
11: -35.4687614, 14.0143147, -35.9780006, 14.1987839, -49.0004044, 49.3378448
12: -49.1666031, 1.6204114, -49.6960144, 1.8936739, -43.7326813, 43.9956970
13: -28.7030716, 21.1595802, -28.7947845, 21.3766747, -49.4736481, 49.3507996
14: -70.6906509, -6.5987778, -71.1257324, -6.4146042, -64.2760468, 64.5269547
15: -17.2309170, 24.5375938, -17.3653908, 24.7565594, -41.9874763, 41.9029846
16: -27.2723846, 23.5081406, -27.5701027, 23.6173859, -48.2353745, 48.3570023
17: -71.0461273, -4.0064926, -71.2835159, -3.8395271, -67.2066040, 67.2770233
18: -34.6319847, 11.6095200, -34.8921356, 11.6630335, -40.4411087, 40.6461334
19: -25.5723648, 5.1773915, -25.7856159, 5.2342806, -29.7795334, 29.9022522
20: -26.3334484, 4.2188888, -26.5644779, 4.3186035, -28.9419479, 29.0620270
21: -31.1509399, 9.9490280, -31.5103760, 10.0724697, -40.0989761, 40.3611679
22: -33.5020447, 6.8239937, -33.6735764, 6.9405942, -38.3316498, 38.4019318
23: -26.7710705, 8.7777224, -27.0028019, 8.8626661, -35.1279068, 35.2191887
24: -23.1264858, 9.8050060, -23.2878113, 9.8558140, -32.6716843, 32.7495422
25: -29.1044731, 5.9730215, -29.2624702, 6.0734296, -34.2723083, 34.2795029
26: -42.7644043, 7.5873895, -43.2085342, 7.7677798, -43.4893341, 43.7705688
27: -26.5119324, 11.4020100, -26.6885433, 11.4563656, -37.5964355, 37.7181778
28: -29.5110073, 7.0603824, -29.6498299, 7.1494431, -36.5312576, 36.5339966
29: -32.5043640, 8.8372059, -32.6680412, 8.9511156, -41.4554787, 41.5052490
30: -37.4467430, 6.7687654, -37.6875153, 6.9161777, -44.3629227, 44.4562798
31: -31.2516270, 7.2296262, -31.5162525, 7.2748275, -37.5120010, 37.6118469
32: -33.5634117, 6.5696354, -33.7770309, 6.7009764, -40.2643890, 40.3466644
33: -43.7557030, 15.8667850, -43.9271469, 16.0999985, -57.0758362, 56.9974060
34: -50.6010742, -4.3989878, -50.7149200, -4.2089338, -42.2515717, 42.1747742
35: -40.8149567, 6.9255872, -40.9504318, 7.1439495, -43.8369980, 43.7562103
36: -44.3911095, 5.3032646, -44.4808159, 5.4174585, -45.4591293, 45.4352875
37: -59.3110352, 2.2954779, -59.4949570, 2.3788090, -55.0591812, 55.1300354
38: -50.8037758, 8.5039701, -50.9417419, 8.6177759, -59.4215508, 59.4457130
39: -52.0283737, 14.8120365, -52.1817741, 14.9242887, -66.9526596, 66.9938126
40: -47.7517395, 8.2557878, -47.8883400, 8.3489332, -53.0686188, 53.1059189
41: -31.7378540, 15.1243439, -31.9161682, 15.2241602, -45.3338165, 45.4408951
42: -27.0382633, 9.9570608, -27.2599182, 10.1150322, -36.5343246, 36.6977730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7058126, upper bound: 14.8146116
time: 43.49 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7348875, upper bound: 14.8146116
time: 22.33 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -23.8060760, 32.8083954, -23.9474888, 33.0326233, -54.3883057, 54.3187103
1: -7.6251888, 32.3235512, -7.6908541, 32.4028397, -36.5063477, 36.5282822
2: -4.6607990, 31.7220287, -4.7591939, 31.9342308, -33.2162933, 33.1095505
3: -8.8509808, 28.8397064, -8.9460735, 29.0872307, -32.5491180, 32.4188156
4: -10.0049305, 35.0098915, -10.1089249, 35.1899834, -43.4301910, 43.3668442
5: -11.0224514, 29.8272076, -11.1395855, 30.1148891, -38.2434616, 38.0809708
6: -38.6602783, 7.4309006, -38.7707329, 7.4913836, -44.3352280, 44.4757614
7: -15.2124319, 30.6112061, -15.3217430, 30.7340565, -41.7474136, 41.7463074
8: -15.3458338, 34.4764290, -15.4445820, 34.6953888, -47.1199646, 47.0138016
9: -10.2643948, 27.0661182, -10.4116497, 27.1725521, -35.7157135, 35.7605667
10: -28.2318363, 23.7119064, -28.6692581, 23.9030457, -50.2928314, 50.5779724
11: -35.5282898, 14.1072025, -35.9903107, 14.2421465, -49.1312866, 49.4505157
12: -49.2206841, 1.7386389, -49.7032700, 1.9480419, -43.8449936, 44.0950317
13: -28.7857704, 21.2145119, -28.8323860, 21.3913231, -49.5771179, 49.4739609
14: -70.8528214, -6.5369644, -71.2045746, -6.4031868, -64.4496307, 64.6676102
15: -17.3108826, 24.5729790, -17.4030170, 24.7656384, -42.0765228, 41.9759979
16: -27.3544064, 23.5387096, -27.6085091, 23.6243477, -48.3248367, 48.4388199
17: -71.0952454, -3.9749775, -71.3043747, -3.8304462, -67.2648010, 67.3293991
18: -34.6716232, 11.6670628, -34.9035339, 11.6907949, -40.5180931, 40.7208176
19: -25.6219196, 5.2263088, -25.7986183, 5.2590680, -29.8623199, 29.9678154
20: -26.3739777, 4.2866974, -26.5721416, 4.3520470, -29.0151596, 29.1264267
21: -31.2124481, 10.0405483, -31.5249748, 10.1182823, -40.2190170, 40.4667969
22: -33.5656052, 6.9218254, -33.6855545, 6.9897866, -38.4452057, 38.4858551
23: -26.8209743, 8.8379498, -27.0143509, 8.8922596, -35.2172852, 35.2941933
24: -23.1789932, 9.8530512, -23.2990189, 9.8794136, -32.7565269, 32.8072433
25: -29.1579857, 6.0536842, -29.2728920, 6.1128092, -34.3659592, 34.3687668
26: -42.8347740, 7.6851540, -43.2199593, 7.8161850, -43.6142044, 43.8551559
27: -26.5685558, 11.4723644, -26.6997795, 11.4906902, -37.7027130, 37.7998314
28: -29.5694962, 7.1412997, -29.6603355, 7.1897788, -36.6303558, 36.6258316
29: -32.5635071, 8.9482365, -32.6781960, 9.0048885, -41.5683975, 41.6264343
30: -37.5091057, 6.8878889, -37.6964493, 6.9746876, -44.4837952, 44.5843391
31: -31.3004990, 7.2628527, -31.5293999, 7.2904844, -37.5968857, 37.6633835
32: -33.6097603, 6.6585989, -33.7896652, 6.7424669, -40.3522263, 40.4482651
33: -43.8335457, 15.9140148, -43.9519806, 16.1227512, -57.1867981, 57.0778427
34: -50.6577072, -4.3160467, -50.7292976, -4.1677785, -42.3566284, 42.2556915
35: -40.8754959, 6.9822659, -40.9648323, 7.1722493, -43.9309540, 43.8282242
36: -44.4422760, 5.3898268, -44.4903183, 5.4615479, -45.5574799, 45.5146713
37: -59.3885880, 2.3378015, -59.5166855, 2.3982053, -55.1618423, 55.1834564
38: -50.8564453, 8.5562582, -50.9538536, 8.6425667, -59.4990120, 59.5101128
39: -52.1077728, 14.8281307, -52.2091446, 14.9323416, -67.0401154, 67.0372772
40: -47.7980766, 8.2909470, -47.9074059, 8.3604918, -53.1296158, 53.1862640
41: -31.7935791, 15.2190876, -31.9304733, 15.2705774, -45.4399261, 45.5459900
42: -27.0780010, 10.0459442, -27.2701874, 10.1572123, -36.6192932, 36.7949409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7727096, upper bound: 14.7855903
time: 42.45 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7727096, upper bound: 14.8146115
time: 84.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.9067612, 32.8961754, -23.9469967, 32.9029694, -54.3443146, 54.3820877
1: -7.6796227, 32.3437958, -7.6987381, 32.3454514, -36.5395813, 36.5125885
2: -4.7471523, 31.8027802, -4.7852936, 31.8049240, -33.1277924, 33.2036819
3: -8.9151621, 28.9175072, -8.9656925, 28.9223728, -32.4061661, 32.4819107
4: -10.1167440, 35.0864487, -10.1483259, 35.0908661, -43.4100266, 43.4523163
5: -11.1213474, 29.9356842, -11.1750517, 29.9425697, -38.1150818, 38.2154388
6: -38.7399750, 7.4732800, -38.7531052, 7.4916191, -44.5052185, 44.4880447
7: -15.2793608, 30.6395111, -15.3163986, 30.6445045, -41.7251816, 41.7457581
8: -15.4551573, 34.5824165, -15.4929838, 34.5868835, -47.0753250, 47.1412506
9: -10.3515320, 27.1944370, -10.3538303, 27.2175751, -35.8528442, 35.7410622
10: -28.4626503, 23.9479446, -28.4634743, 24.0156403, -50.6698837, 50.4037857
11: -35.7903824, 14.2925024, -35.8035927, 14.3419552, -49.4755402, 49.3834381
12: -49.4179306, 1.9604063, -49.4237518, 2.0557890, -44.1266479, 44.0010529
13: -28.8141232, 21.3336658, -28.8261490, 21.3547325, -49.6013031, 49.5739822
14: -71.0012894, -6.3835831, -71.0079117, -6.3229542, -64.6783371, 64.6243286
15: -17.4202023, 24.6891594, -17.4412575, 24.6999512, -42.1201553, 42.1304169
16: -27.5243855, 23.6770077, -27.5337620, 23.7026119, -48.5498657, 48.4383240
17: -71.1639023, -3.8279228, -71.1716156, -3.7644024, -67.3994980, 67.3436890
18: -34.7534332, 11.6645432, -34.7605591, 11.6876183, -40.5932922, 40.5888252
19: -25.7077847, 5.2647142, -25.7168007, 5.2840643, -29.9261780, 29.9410400
20: -26.4761162, 4.3603110, -26.4842777, 4.3836184, -29.1042175, 29.0518036
21: -31.3659763, 10.1338520, -31.3768215, 10.1716337, -40.4328690, 40.3094254
22: -33.6181526, 6.9684925, -33.6336021, 6.9964137, -38.4789886, 38.4908829
23: -26.9150543, 8.8948917, -26.9217491, 8.9089766, -35.2491455, 35.2534714
24: -23.2614708, 9.8665876, -23.2750511, 9.8746614, -32.8237724, 32.8577347
25: -29.2171059, 6.1080499, -29.2259617, 6.1285925, -34.3874130, 34.4091492
26: -42.9552879, 7.7712426, -42.9667320, 7.8468156, -43.7575073, 43.6712799
27: -26.6495018, 11.4755955, -26.6680374, 11.4880886, -37.7797241, 37.7964706
28: -29.6063633, 7.1700206, -29.6143684, 7.1774783, -36.6304321, 36.6595306
29: -32.6095352, 8.9983902, -32.6160736, 9.0280437, -41.6375809, 41.6144638
30: -37.5932121, 6.9708614, -37.6031189, 6.9933062, -44.5865173, 44.5739822
31: -31.4139690, 7.2909060, -31.4244785, 7.3033714, -37.6074371, 37.7145729
32: -33.7205544, 6.7587681, -33.7262077, 6.7786989, -40.4992523, 40.4849777
33: -43.9679832, 16.0228596, -44.0095482, 16.0283508, -57.1532745, 57.2509537
34: -50.7377167, -4.2446218, -50.7611046, -4.2413492, -42.2984619, 42.3829880
35: -40.9854736, 7.0794740, -41.0176849, 7.0816836, -43.8952332, 43.9858932
36: -44.4895172, 5.4351082, -44.4991112, 5.4471073, -45.5761108, 45.5909042
37: -59.4676056, 2.3794732, -59.4815598, 2.3901596, -55.2117233, 55.2459335
38: -50.9384193, 8.6215906, -50.9538841, 8.6368599, -59.5752792, 59.5754738
39: -52.1830750, 14.9067125, -52.1975174, 14.9216385, -67.1047134, 67.1042328
40: -47.8896637, 8.3266478, -47.9085693, 8.3323812, -53.2057037, 53.2023926
41: -31.9010963, 15.2686577, -31.9135265, 15.2779140, -45.5592575, 45.5649338
42: -27.1982079, 10.1860399, -27.2068119, 10.2080164, -36.8563919, 36.8317795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
time: 34.79 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
time: 45.74 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.9946022, 33.0477524, -23.9504070, 32.9034042, -54.4341125, 54.5460968
1: -7.7135229, 32.4041214, -7.6984081, 32.3451767, -36.5812225, 36.5726814
2: -4.8114967, 31.9357414, -4.7917910, 31.8044891, -33.1902237, 33.3501740
3: -8.9892559, 29.0961952, -8.9734297, 28.9218655, -32.4719620, 32.6675644
4: -10.1625376, 35.1947746, -10.1514874, 35.0903931, -43.4580994, 43.5685272
5: -11.2045059, 30.1283417, -11.1843395, 29.9423752, -38.1949310, 38.4194183
6: -38.7949638, 7.4921455, -38.7530861, 7.4883099, -44.6043701, 44.5063324
7: -15.3538866, 30.7401772, -15.3187904, 30.6432114, -41.7960510, 41.8562012
8: -15.5076008, 34.7041168, -15.4956169, 34.5866394, -47.1285095, 47.2814331
9: -10.4146252, 27.2317810, -10.3529472, 27.2161617, -35.9115601, 35.7835236
10: -28.6776085, 24.0335541, -28.4635677, 24.0226116, -50.8976517, 50.4875717
11: -36.0141602, 14.3452578, -35.8034058, 14.3469067, -49.7172241, 49.4381485
12: -49.7116814, 2.0891094, -49.4223175, 2.0674210, -44.4333801, 44.1187592
13: -28.8375950, 21.4299164, -28.8223667, 21.3561802, -49.6248703, 49.6617737
14: -71.2126083, -6.3065910, -71.0068512, -6.3135376, -64.8990707, 64.7002563
15: -17.4439430, 24.7848797, -17.4340057, 24.7001286, -42.1440735, 42.2188873
16: -27.6251392, 23.6895485, -27.5331936, 23.6990662, -48.6205444, 48.4491730
17: -71.3143616, -3.7425995, -71.1706238, -3.7545834, -67.5597763, 67.4280243
18: -34.9191780, 11.6960335, -34.7603378, 11.6876421, -40.7725296, 40.6344452
19: -25.8150635, 5.2856507, -25.7171745, 5.2862043, -30.0183640, 29.9570465
20: -26.5867786, 4.3919930, -26.4848099, 4.3859172, -29.2319946, 29.0861549
21: -31.5411186, 10.1766968, -31.3768158, 10.1763802, -40.6242981, 40.3551102
22: -33.6937790, 7.0139275, -33.6322594, 6.9939647, -38.5601959, 38.5350723
23: -27.0242786, 8.9199924, -26.9218330, 8.9103680, -35.3489914, 35.2834702
24: -23.3219376, 9.8818550, -23.2751255, 9.8723888, -32.8762741, 32.8752289
25: -29.2860470, 6.1417170, -29.2255535, 6.1304660, -34.4474335, 34.4433098
26: -43.2386284, 7.8865762, -42.9655609, 7.8571882, -44.0538559, 43.7765884
27: -26.7251015, 11.4932909, -26.6687546, 11.4871140, -37.8494797, 37.8153801
28: -29.6727791, 7.1962748, -29.6146889, 7.1767149, -36.6852188, 36.7007980
29: -32.6864586, 9.0362682, -32.6148453, 9.0298624, -41.7163200, 41.6511154
30: -37.7115860, 7.0118275, -37.6030350, 6.9948912, -44.7064781, 44.6148605
31: -31.5487442, 7.3068099, -31.4251118, 7.3044176, -37.7095146, 37.7429504
32: -33.8009148, 6.7877626, -33.7258606, 6.7778978, -40.5788116, 40.5136223
33: -44.0222321, 16.1362991, -44.0135803, 16.0280762, -57.2071686, 57.3823547
34: -50.7699738, -4.1621180, -50.7632256, -4.2414465, -42.3280106, 42.4685211
35: -41.0228767, 7.1805286, -41.0204468, 7.0818839, -43.9319534, 44.0884857
36: -44.5064011, 5.4737167, -44.4984856, 5.4453526, -45.5952530, 45.6337357
37: -59.5433846, 2.4113760, -59.4813118, 2.3860393, -55.2811890, 55.2972717
38: -50.9843521, 8.6637125, -50.9533844, 8.6345730, -59.6189270, 59.6170959
39: -52.2344208, 14.9577084, -52.1978836, 14.9177914, -67.1522141, 67.1555939
40: -47.9422455, 8.3660183, -47.9090729, 8.3306808, -53.2595367, 53.2449417
41: -31.9562035, 15.2914133, -31.9136620, 15.2749357, -45.6290512, 45.5849380
42: -27.2837143, 10.2169037, -27.2066765, 10.2031326, -36.9973221, 36.8694458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7855901, upper bound: 14.8146118
time: 39.51 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8146113, upper bound: 14.8146118
time: 36.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 77.84 seconds
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 77.84
Output dim: 2, lower bound: -14.7058126, upper bound: 14.8146116
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 77.84
Output dim: 2, lower bound: -14.7348875, upper bound: 14.8146116
IS_A1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 77.84
Output dim: 2, lower bound: -14.7727096, upper bound: 14.7855903
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 77.84
Output dim: 2, lower bound: -14.7727096, upper bound: 14.8146115
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 77.84
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 77.84
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 77.84
Output dim: 2, lower bound: -14.7855901, upper bound: 14.8146118
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 77.84
Output dim: 2, lower bound: -14.8146113, upper bound: 14.8146118

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -23.6342945, 32.7289658, -23.8523102, 32.9959831, -54.2059402, 54.1462708
1: -7.5381794, 32.2729301, -7.6435547, 32.3672104, -36.4072342, 36.4310989
2: -4.5431900, 31.6525879, -4.6945610, 31.8841534, -33.0678329, 32.9750519
3: -8.7685719, 28.7848873, -8.9002914, 29.0676689, -32.4604721, 32.3163071
4: -9.8759174, 34.9304161, -10.0409422, 35.1391258, -43.2561493, 43.2175751
5: -10.9362097, 29.7573242, -11.0920515, 30.0857697, -38.1382904, 37.9607773
6: -38.5790787, 7.2797298, -38.6879349, 7.4119396, -44.1684494, 44.2470932
7: -15.1103401, 30.5523186, -15.2662487, 30.6973686, -41.6342010, 41.6306076
8: -15.1889381, 34.3785667, -15.3626652, 34.6431999, -46.9322205, 46.8330765
9: -10.1744719, 27.0240574, -10.3579340, 27.1584778, -35.5996857, 35.6421089
10: -28.1552048, 23.6437950, -28.6257038, 23.8418236, -50.1311798, 50.4025574
11: -35.4648666, 14.0003805, -35.9690170, 14.1671753, -48.9644928, 49.3142014
12: -49.1534004, 1.6160622, -49.6657104, 1.8836160, -43.7043533, 43.9540482
13: -28.6977768, 21.1550331, -28.7827034, 21.3662109, -49.4537659, 49.3318405
14: -70.6855621, -6.6308784, -71.1139450, -6.4890347, -64.1965256, 64.4830627
15: -17.2282295, 24.5320568, -17.3591957, 24.7438126, -41.9720421, 41.8912506
16: -27.2690334, 23.5015697, -27.5623531, 23.6025009, -48.2115479, 48.3400955
17: -71.0382309, -4.0563736, -71.2654266, -3.9550457, -67.0831833, 67.2090530
18: -34.6279144, 11.6016340, -34.8827667, 11.6456604, -40.4205933, 40.6291542
19: -25.5685654, 5.1708488, -25.7768173, 5.2192841, -29.7619247, 29.8868294
20: -26.3303871, 4.2163410, -26.5573788, 4.3126965, -28.9325104, 29.0521774
21: -31.1473255, 9.9421253, -31.5020428, 10.0566998, -40.0816650, 40.3462753
22: -33.4981918, 6.8186064, -33.6646996, 6.9282293, -38.3062668, 38.3821487
23: -26.7677193, 8.7713404, -26.9949951, 8.8480453, -35.1109772, 35.2048798
24: -23.1216507, 9.7938404, -23.2767124, 9.8301105, -32.6412544, 32.7269821
25: -29.1012535, 5.9617543, -29.2550831, 6.0477157, -34.2425003, 34.2600174
26: -42.7596054, 7.5847387, -43.1975250, 7.7616730, -43.4766922, 43.7544327
27: -26.4964161, 11.3999510, -26.6525917, 11.4516726, -37.5735245, 37.6764679
28: -29.5047836, 7.0567927, -29.6356506, 7.1412158, -36.5142517, 36.5135498
29: -32.4988708, 8.8325424, -32.6554489, 8.9404011, -41.4392700, 41.4879913
30: -37.4445534, 6.7624397, -37.6824875, 6.9016542, -44.3462067, 44.4449272
31: -31.2465267, 7.2155066, -31.5044632, 7.2423954, -37.4771194, 37.5866737
32: -33.5294304, 6.5665283, -33.6983414, 6.6938181, -40.2232475, 40.2648697
33: -43.7485771, 15.8645029, -43.9108200, 16.0946541, -57.0582352, 56.9719162
34: -50.5768700, -4.4013672, -50.6589928, -4.2143488, -42.2219620, 42.1170883
35: -40.8010597, 6.9244723, -40.9187393, 7.1414261, -43.8202591, 43.7243271
36: -44.3561096, 5.3012881, -44.3998947, 5.4129286, -45.4194565, 45.3520050
37: -59.2915421, 2.2911062, -59.4513512, 2.3686495, -55.0271454, 55.0754471
38: -50.7734642, 8.5012770, -50.8724365, 8.6114273, -59.3848915, 59.3737144
39: -52.0143051, 14.8093691, -52.1494522, 14.9180775, -66.9323807, 66.9588242
40: -47.7236824, 8.2525873, -47.8236542, 8.3415604, -53.0317078, 53.0359802
41: -31.7059517, 15.1213722, -31.8423500, 15.2172823, -45.2948608, 45.3638382
42: -27.0158424, 9.9538841, -27.2082481, 10.1077232, -36.5043869, 36.6426735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6593511, upper bound: 14.8130235
time: 39.05 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7042234, upper bound: 14.8130235
time: 45.31 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -23.6364555, 32.7362022, -23.9259510, 33.0259972, -54.2392654, 54.2269669
1: -7.5391245, 32.2809563, -7.6947899, 32.3948479, -36.4322739, 36.4919434
2: -4.5450521, 31.6684494, -4.7872105, 31.9328480, -33.1159210, 33.0835419
3: -8.7704964, 28.7872562, -8.9482994, 29.0840225, -32.4762878, 32.3697205
4: -9.8775787, 34.9457512, -10.1329670, 35.1890450, -43.3066406, 43.3238144
5: -10.9380932, 29.7615318, -11.1382160, 30.1082020, -38.1605453, 38.0201797
6: -38.6066704, 7.2811098, -38.7757988, 7.5460634, -44.3296509, 44.3313370
7: -15.1119137, 30.5635090, -15.3266268, 30.7322388, -41.6695328, 41.7025757
8: -15.1909332, 34.3938065, -15.4446936, 34.6931343, -46.9809113, 46.9306107
9: -10.1718473, 27.0259628, -10.3867702, 27.1924782, -35.6337051, 35.6775894
10: -28.1572971, 23.6552353, -28.7048931, 23.8936806, -50.1847916, 50.5339966
11: -35.4662247, 14.0099077, -36.0165901, 14.2069817, -49.0067596, 49.3703156
12: -49.1633415, 1.6178751, -49.7041054, 1.9275551, -43.7651138, 43.9980469
13: -28.7007942, 21.1554337, -28.8241177, 21.3936157, -49.4812698, 49.3836441
14: -70.6876221, -6.6055603, -71.2552185, -6.4136734, -64.2739487, 64.6496582
15: -17.2296295, 24.5342274, -17.3929176, 24.7604923, -41.9901199, 41.9271469
16: -27.2701855, 23.5046959, -27.6200199, 23.6290379, -48.2403183, 48.4019852
17: -71.0431213, -4.0162468, -71.4563751, -3.8419781, -67.2011414, 67.4401245
18: -34.6295204, 11.6001225, -34.9256859, 11.6693945, -40.4510269, 40.6682358
19: -25.5701256, 5.1739721, -25.8104191, 5.2423792, -29.7851448, 29.9176941
20: -26.3305073, 4.2175961, -26.5678482, 4.3358231, -28.9543533, 29.0672684
21: -31.1481457, 9.9448814, -31.5323753, 10.0801153, -40.1055374, 40.3740540
22: -33.5001602, 6.8213964, -33.6998215, 6.9534254, -38.3321075, 38.4397964
23: -26.7689419, 8.7736702, -27.0215225, 8.8723898, -35.1373444, 35.2328949
24: -23.1227856, 9.7994995, -23.3335419, 9.8663597, -32.6773911, 32.7876167
25: -29.1016197, 5.9665279, -29.3040810, 6.0825558, -34.2794724, 34.3199806
26: -42.7612762, 7.5840564, -43.2233353, 7.7747345, -43.4974060, 43.7855988
27: -26.5051632, 11.4008465, -26.6986561, 11.5156498, -37.6666946, 37.7267647
28: -29.5055313, 7.0589366, -29.6541157, 7.1836891, -36.5626907, 36.5342789
29: -32.4998703, 8.8354845, -32.6865997, 8.9652424, -41.4651108, 41.5220833
30: -37.4444351, 6.7651939, -37.6969566, 6.9302540, -44.3746872, 44.4621506
31: -31.2488861, 7.2254362, -31.5431843, 7.2828689, -37.5138702, 37.6263199
32: -33.5566864, 6.5673008, -33.7886276, 6.8307114, -40.3873978, 40.3559265
33: -43.7511749, 15.8658333, -43.9350090, 16.1384163, -57.1187134, 57.0047531
34: -50.5938492, -4.4005637, -50.7208519, -4.1236906, -42.3381882, 42.1797104
35: -40.8096390, 6.9251957, -40.9561386, 7.1890655, -43.8838806, 43.7603073
36: -44.3838539, 5.3020468, -44.4877319, 5.5358772, -45.5703430, 45.4397507
37: -59.3024712, 2.2927065, -59.4984512, 2.4517226, -55.1317825, 55.1248779
38: -50.7943497, 8.5026054, -50.9536400, 8.7164173, -59.5107651, 59.4562454
39: -52.0220261, 14.8102922, -52.1937943, 14.9662857, -66.9883118, 67.0040894
40: -47.7456970, 8.2539454, -47.8957977, 8.4600029, -53.1722107, 53.1076431
41: -31.7310448, 15.1224699, -31.9285297, 15.3505459, -45.4543762, 45.4490814
42: -27.0335178, 9.9553070, -27.2707348, 10.2080193, -36.6222153, 36.7033157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7042235, upper bound: 14.7682098
time: 42.00 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7332977, upper bound: 14.8130234
time: 35.04 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.8688660, 32.8075409, -23.9450054, 33.0265236, -54.4445038, 54.3165131
1: -7.6710787, 32.3211632, -7.6894712, 32.3977852, -36.5491409, 36.5213127
2: -4.7461224, 31.7257538, -4.7578764, 31.9302597, -33.2978897, 33.1091576
3: -8.8914232, 28.8438530, -8.9446468, 29.0843086, -32.5878525, 32.4165192
4: -10.0903168, 35.0149460, -10.1076460, 35.1859055, -43.5100327, 43.3691101
5: -11.0615463, 29.8302975, -11.1384144, 30.1106987, -38.2852020, 38.0770493
6: -38.6737137, 7.5586023, -38.7657204, 7.4899750, -44.3417740, 44.5971680
7: -15.2665100, 30.6134872, -15.3206081, 30.7309570, -41.7988358, 41.7454910
8: -15.4200249, 34.4832840, -15.4431877, 34.6920204, -47.1903763, 47.0158615
9: -10.2802124, 27.0909157, -10.4034004, 27.1704674, -35.7347336, 35.7780037
10: -28.3040524, 23.7147484, -28.6683331, 23.8932190, -50.3892136, 50.5671082
11: -35.5669632, 14.1152554, -35.9878082, 14.2377167, -49.1637268, 49.4568253
12: -49.2288742, 1.7725282, -49.6999893, 1.9454699, -43.8473740, 44.1275101
13: -28.8151493, 21.2315102, -28.8301353, 21.3871994, -49.6100006, 49.4815750
14: -70.9822540, -6.5361214, -71.2015533, -6.4099178, -64.5723343, 64.6654358
15: -17.3384438, 24.5769024, -17.4017277, 24.7622414, -42.1006851, 41.9786301
16: -27.4042435, 23.5503349, -27.6063137, 23.6208992, -48.3697815, 48.4438019
17: -71.2679977, -3.9774818, -71.3013229, -3.8401756, -67.4278259, 67.3238373
18: -34.7051392, 11.6734562, -34.9010773, 11.6814280, -40.5400772, 40.7308960
19: -25.6466064, 5.2343292, -25.7963753, 5.2556829, -29.8777237, 29.9734344
20: -26.3773708, 4.3039174, -26.5692158, 4.3507318, -29.0204239, 29.1387482
21: -31.2345085, 10.0481863, -31.5221748, 10.1141090, -40.2319489, 40.4733658
22: -33.5917931, 6.9345431, -33.6836319, 6.9871683, -38.4830322, 38.4862518
23: -26.8396721, 8.8475914, -27.0121689, 8.8882141, -35.2309341, 35.3036957
24: -23.2247524, 9.8635235, -23.2953472, 9.8739309, -32.7946663, 32.8129654
25: -29.1995659, 6.0626945, -29.2700310, 6.1063180, -34.4064178, 34.3759117
26: -42.8494949, 7.6921120, -43.2168312, 7.8128619, -43.6292419, 43.8631363
27: -26.5786552, 11.5316238, -26.6930313, 11.4895353, -37.7113037, 37.8700714
28: -29.5738583, 7.1755872, -29.6548195, 7.1883149, -36.6306763, 36.6574860
29: -32.5820160, 8.9622879, -32.6737556, 9.0031738, -41.5851898, 41.6360435
30: -37.5186386, 6.9018936, -37.6941376, 6.9710770, -44.4897156, 44.5960312
31: -31.3273411, 7.2708445, -31.5266628, 7.2863121, -37.6112404, 37.6652374
32: -33.6214752, 6.7883816, -33.7829056, 6.7401056, -40.3615799, 40.5712891
33: -43.8413734, 15.9525452, -43.9475212, 16.1218147, -57.1940842, 57.1207123
34: -50.6636581, -4.2308898, -50.7220993, -4.1693325, -42.3615036, 42.3422012
35: -40.8811455, 7.0274019, -40.9594955, 7.1718693, -43.9350433, 43.8750839
36: -44.4491501, 5.5082526, -44.4830475, 5.4603748, -45.5618896, 45.6259079
37: -59.3918762, 2.4107904, -59.5081139, 2.3954358, -55.1567078, 55.2560959
38: -50.8678360, 8.6549206, -50.9444733, 8.6412010, -59.5090370, 59.5993958
39: -52.1198921, 14.8700943, -52.2027512, 14.9305325, -67.0504227, 67.0728455
40: -47.8055115, 8.4019613, -47.9013290, 8.3586054, -53.1312637, 53.2899094
41: -31.8059731, 15.3454790, -31.9237022, 15.2686634, -45.4481659, 45.6664886
42: -27.0889149, 10.1389914, -27.2654648, 10.1554813, -36.6249237, 36.8828812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7262842, upper bound: 14.8130234
time: 44.49 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6884390, upper bound: 14.8130234
time: 34.30 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -23.9021034, 32.8828011, -23.9362659, 32.8720245, -54.3086548, 54.3575668
1: -7.6773157, 32.3306847, -7.6934090, 32.3153572, -36.5076828, 36.4945450
2: -4.7439499, 31.7829628, -4.7779713, 31.7599525, -33.0792923, 33.1768494
3: -8.9118690, 28.9121685, -8.9581242, 28.9101791, -32.3880615, 32.4672966
4: -10.1138306, 35.0670509, -10.1416092, 35.0459213, -43.3617554, 43.4259186
5: -11.1182690, 29.9272919, -11.1680775, 29.9231777, -38.0888519, 38.1978302
6: -38.7073441, 7.4705067, -38.6787109, 7.4852295, -44.4654999, 44.4101562
7: -15.2766809, 30.6252079, -15.3102036, 30.6119213, -41.6890259, 41.7252960
8: -15.4518185, 34.5638123, -15.4852018, 34.5437698, -47.0287323, 47.1141510
9: -10.3458729, 27.1904259, -10.3408070, 27.2082672, -35.8362503, 35.7244949
10: -28.4596214, 23.9267063, -28.4564819, 23.9666061, -50.6054230, 50.3688431
11: -35.7864265, 14.2786102, -35.7946320, 14.3103323, -49.4396820, 49.3598328
12: -49.4047165, 1.9560542, -49.3934479, 2.0457273, -44.0984421, 43.9594040
13: -28.8088379, 21.3291435, -28.8141193, 21.3442039, -49.5813446, 49.5550232
14: -70.9961243, -6.4156799, -70.9962387, -6.3973999, -64.5987244, 64.5805588
15: -17.4174843, 24.6835976, -17.4350662, 24.6872005, -42.1046829, 42.1186638
16: -27.5210152, 23.6704521, -27.5260258, 23.6878014, -48.5260620, 48.4213715
17: -71.1559677, -3.8777962, -71.1534882, -3.8799324, -67.2760315, 67.2756958
18: -34.7493477, 11.6566849, -34.7511330, 11.6703043, -40.5727921, 40.5718155
19: -25.7039757, 5.2581949, -25.7080040, 5.2690983, -29.9085846, 29.9255714
20: -26.4730434, 4.3577361, -26.4771614, 4.3777299, -29.0947571, 29.0419998
21: -31.3623657, 10.1269875, -31.3685169, 10.1559296, -40.4155197, 40.2945099
22: -33.6142502, 6.9631124, -33.6246834, 6.9840331, -38.4535904, 38.4710770
23: -26.9116707, 8.8885670, -26.9139805, 8.8943462, -35.2322540, 35.2391014
24: -23.2566528, 9.8554192, -23.2639084, 9.8489666, -32.7933578, 32.8351364
25: -29.2138901, 6.0967546, -29.2186241, 6.1029072, -34.3576660, 34.3896790
26: -42.9504929, 7.7685633, -42.9556541, 7.8406696, -43.7448425, 43.6552048
27: -26.6339283, 11.4735355, -26.6320648, 11.4833784, -37.7568359, 37.7546768
28: -29.6001053, 7.1664491, -29.6001530, 7.1692448, -36.6134033, 36.6389542
29: -32.6040268, 8.9937553, -32.6034088, 9.0173416, -41.6213684, 41.5971642
30: -37.5910110, 6.9645319, -37.5980835, 6.9787998, -44.5698090, 44.5626144
31: -31.4088650, 7.2768130, -31.4126816, 7.2710028, -37.5726509, 37.6893578
32: -33.6865883, 6.7556705, -33.6475410, 6.7715607, -40.4581490, 40.4032135
33: -43.9608917, 16.0205078, -43.9931984, 16.0230427, -57.1357193, 57.2253494
34: -50.7135391, -4.2470360, -50.7051506, -4.2468019, -42.2688217, 42.3253021
35: -40.9715576, 7.0783653, -40.9859161, 7.0790963, -43.8784485, 43.9540405
36: -44.4544907, 5.4331846, -44.4182281, 5.4425554, -45.5364838, 45.5076218
37: -59.4480286, 2.3750491, -59.4378586, 2.3799982, -55.1796646, 55.1912766
38: -50.9081383, 8.6188259, -50.8843994, 8.6305380, -59.5386772, 59.5032272
39: -52.1690025, 14.9040499, -52.1653137, 14.9154139, -67.0844193, 67.0693665
40: -47.8616409, 8.3234348, -47.8439674, 8.3250341, -53.1688156, 53.1325150
41: -31.8692398, 15.2656660, -31.8397446, 15.2710495, -45.5203400, 45.4878693
42: -27.1758270, 10.1828651, -27.1551037, 10.2007713, -36.8265076, 36.7766495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237
time: 39.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 28.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -23.9042625, 32.8900833, -24.0098267, 32.9020500, -54.3420105, 54.4382935
1: -7.6782637, 32.3387337, -7.7446070, 32.3430595, -36.5326385, 36.5553970
2: -4.7458344, 31.7988434, -4.8705883, 31.8086758, -33.1274109, 33.2852859
3: -8.9137449, 28.9145584, -9.0061960, 28.9265976, -32.4038849, 32.5207901
4: -10.1154690, 35.0823975, -10.2336712, 35.0959129, -43.4122849, 43.5321960
5: -11.1201439, 29.9315224, -11.2142076, 29.9456367, -38.1111298, 38.2572098
6: -38.7349396, 7.4718800, -38.7665215, 7.6193085, -44.6266785, 44.4943848
7: -15.2782593, 30.6364098, -15.3705826, 30.6467953, -41.7244034, 41.7972794
8: -15.4537601, 34.5790024, -15.5672779, 34.5937424, -47.0774841, 47.2117538
9: -10.3432951, 27.1923580, -10.3696299, 27.2423172, -35.8703079, 35.7599945
10: -28.4617004, 23.9381599, -28.5356522, 24.0184517, -50.6590347, 50.5002289
11: -35.7878723, 14.2881050, -35.8422585, 14.3500814, -49.4818420, 49.4159546
12: -49.4146309, 1.9578805, -49.4318733, 2.0896254, -44.1590881, 44.0033875
13: -28.8118286, 21.3295555, -28.8556175, 21.3717651, -49.6089172, 49.6068726
14: -70.9982300, -6.3903427, -71.1374054, -6.3220673, -64.6761627, 64.7470627
15: -17.4189491, 24.6857567, -17.4687996, 24.7038708, -42.1228180, 42.1545563
16: -27.5221672, 23.6735344, -27.5836201, 23.7144012, -48.5548859, 48.4832840
17: -71.1608505, -3.8376808, -71.3443909, -3.7668190, -67.3940277, 67.5067139
18: -34.7509689, 11.6551771, -34.7940979, 11.6939802, -40.6033020, 40.6107597
19: -25.7055168, 5.2613120, -25.7415161, 5.2921696, -29.9318390, 29.9563980
20: -26.4731979, 4.3589725, -26.4876423, 4.4008298, -29.1165810, 29.0570488
21: -31.3631783, 10.1296806, -31.3988037, 10.1793203, -40.4394379, 40.3223572
22: -33.6162148, 6.9658861, -33.6597443, 7.0091491, -38.4794922, 38.5286713
23: -26.9129505, 8.8908405, -26.9405174, 8.9186420, -35.2586517, 35.2671051
24: -23.2577991, 9.8610992, -23.3206654, 9.8851051, -32.8294067, 32.8957214
25: -29.2142620, 6.1015587, -29.2675285, 6.1376872, -34.3945770, 34.4495697
26: -42.9522209, 7.7679124, -42.9814796, 7.8538399, -43.7655487, 43.6863861
27: -26.6427288, 11.4744329, -26.6782112, 11.5473518, -37.8500137, 37.8051224
28: -29.6008224, 7.1685543, -29.6187057, 7.2117786, -36.6619034, 36.6597977
29: -32.6050873, 8.9966669, -32.6346207, 9.0421467, -41.6472321, 41.6312866
30: -37.5909042, 6.9672565, -37.6126175, 7.0073862, -44.5982895, 44.5798721
31: -31.4112110, 7.2867222, -31.4513588, 7.3113956, -37.6093292, 37.7288475
32: -33.7138100, 6.7564554, -33.7378082, 6.9083681, -40.6221771, 40.4942627
33: -43.9634666, 16.0218658, -44.0173607, 16.0668011, -57.1963577, 57.2581940
34: -50.7304611, -4.2462192, -50.7670135, -4.1561551, -42.3849640, 42.3879089
35: -40.9801636, 7.0790739, -41.0233650, 7.1267962, -43.9420776, 43.9899597
36: -44.4822235, 5.4338942, -44.5059967, 5.5655026, -45.6873169, 45.5953445
37: -59.4590454, 2.3766942, -59.4847946, 2.4631062, -55.2843552, 55.2408066
38: -50.9289856, 8.6201849, -50.9653397, 8.7354259, -59.6644135, 59.5855255
39: -52.1767197, 14.9049387, -52.2096863, 14.9635773, -67.1402969, 67.1146240
40: -47.8836288, 8.3247528, -47.9160614, 8.4434700, -53.3092957, 53.2041550
41: -31.8943501, 15.2667789, -31.9259567, 15.4043312, -45.6797333, 45.5731583
42: -27.1934490, 10.1842823, -27.2176437, 10.3010798, -36.9443054, 36.8373299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7109163, upper bound: 14.7682100
time: 33.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7109163, upper bound: 14.8130236
time: 39.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -23.9899216, 33.0343704, -23.9397087, 32.8724365, -54.3983612, 54.5215988
1: -7.7111683, 32.3910027, -7.6930780, 32.3151093, -36.5493088, 36.5546227
2: -4.8083062, 31.9159241, -4.7844491, 31.7594948, -33.1417313, 33.3233185
3: -8.9859467, 29.0909042, -8.9658546, 28.9096375, -32.4538193, 32.6529427
4: -10.1596518, 35.1753426, -10.1447964, 35.0454865, -43.4097977, 43.5420837
5: -11.2014828, 30.1199741, -11.1773338, 29.9230022, -38.1686096, 38.4017792
6: -38.7623711, 7.4893436, -38.6786537, 7.4819317, -44.5646362, 44.4284286
7: -15.3511572, 30.7258835, -15.3126097, 30.6106606, -41.7598877, 41.8357086
8: -15.5042267, 34.6855621, -15.4878263, 34.5435410, -47.0818787, 47.2543106
9: -10.4089403, 27.2277775, -10.3399124, 27.2069244, -35.8949661, 35.7669792
10: -28.6745777, 24.0123291, -28.4565773, 23.9736099, -50.8332214, 50.4526443
11: -36.0102654, 14.3313227, -35.7944527, 14.3152609, -49.6814270, 49.4145813
12: -49.6985016, 2.0847020, -49.3920250, 2.0574160, -44.4051819, 44.0771179
13: -28.8323250, 21.4253292, -28.8102913, 21.3456497, -49.6048508, 49.6428070
14: -71.2074203, -6.3387184, -70.9951248, -6.3880043, -64.8194122, 64.6564026
15: -17.4412231, 24.7792950, -17.4277992, 24.6873474, -42.1285706, 42.2070923
16: -27.6217461, 23.6829777, -27.5254002, 23.6842098, -48.5967560, 48.4322815
17: -71.3064880, -3.7924290, -71.1524582, -3.8701019, -67.4363861, 67.3600311
18: -34.9151001, 11.6881886, -34.7509537, 11.6703072, -40.7520332, 40.6174316
19: -25.8112526, 5.2791252, -25.7083797, 5.2712450, -30.0007629, 29.9416466
20: -26.5836983, 4.3894291, -26.4776955, 4.3800106, -29.2225342, 29.0763206
21: -31.5375042, 10.1698284, -31.3684750, 10.1606579, -40.6069794, 40.3402405
22: -33.6899109, 7.0085039, -33.6233444, 6.9815540, -38.5347748, 38.5152969
23: -27.0208855, 8.9136362, -26.9140434, 8.8957729, -35.3321152, 35.2691040
24: -23.3171082, 9.8707161, -23.2639771, 9.8466759, -32.8458176, 32.8526459
25: -29.2828255, 6.1304302, -29.2181778, 6.1047993, -34.4176483, 34.4238205
26: -43.2338371, 7.8839669, -42.9545059, 7.8510728, -44.0411682, 43.7604675
27: -26.7096043, 11.4912281, -26.6327534, 11.4824009, -37.8265457, 37.7736320
28: -29.6665154, 7.1926847, -29.6004906, 7.1684799, -36.6683121, 36.6802139
29: -32.6809464, 9.0316648, -32.6021957, 9.0191355, -41.7000809, 41.6338615
30: -37.7093849, 7.0054722, -37.5979729, 6.9803581, -44.6897430, 44.6034470
31: -31.5436192, 7.2927179, -31.4133301, 7.2720571, -37.6746979, 37.7177277
32: -33.7669792, 6.7846527, -33.6472015, 6.7707605, -40.5377388, 40.4318542
33: -44.0151482, 16.1339626, -43.9972610, 16.0228043, -57.1895294, 57.3568649
34: -50.7458344, -4.1644559, -50.7072983, -4.2469540, -42.2984085, 42.4108505
35: -41.0089760, 7.1794248, -40.9887543, 7.0793262, -43.9151611, 44.0566406
36: -44.4714127, 5.4717441, -44.4175758, 5.4408360, -45.5555878, 45.5504379
37: -59.5239029, 2.4069519, -59.4376030, 2.3759089, -55.2491226, 55.2425919
38: -50.9542542, 8.6609373, -50.8838272, 8.6282597, -59.5825119, 59.5447655
39: -52.2203140, 14.9549580, -52.1657295, 14.9115877, -67.1319046, 67.1206894
40: -47.9142113, 8.3628092, -47.8444290, 8.3233337, -53.2226944, 53.1750793
41: -31.9243317, 15.2884369, -31.8398533, 15.2680931, -45.5900650, 45.5078888
42: -27.2613583, 10.2137356, -27.1549854, 10.1958370, -36.9674683, 36.8143234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7840025, upper bound: 14.7682100
time: 44.09 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7840025, upper bound: 14.8130236
time: 38.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.9921150, 33.0416412, -24.0132446, 32.9024811, -54.4317703, 54.6023483
1: -7.7121477, 32.3990669, -7.7442579, 32.3427811, -36.5742874, 36.6154671
2: -4.8101730, 31.9317989, -4.8770747, 31.8082161, -33.1898499, 33.4317551
3: -8.9878502, 29.0932407, -9.0139189, 28.9260063, -32.4696655, 32.7064743
4: -10.1612949, 35.1906929, -10.2368402, 35.0954666, -43.4602966, 43.6483688
5: -11.2033510, 30.1241875, -11.2234831, 29.9454536, -38.1909637, 38.4611969
6: -38.7899590, 7.4907188, -38.7664642, 7.6160011, -44.7257690, 44.5126724
7: -15.3527737, 30.7370491, -15.3729296, 30.6454945, -41.7952957, 41.9076996
8: -15.5062037, 34.7007599, -15.5698805, 34.5935173, -47.1306076, 47.3519363
9: -10.4063749, 27.2296982, -10.3687267, 27.2409801, -35.9289856, 35.8024597
10: -28.6767006, 24.0237617, -28.5357800, 24.0254288, -50.8868103, 50.5839844
11: -36.0116768, 14.3408470, -35.8420715, 14.3550558, -49.7235413, 49.4706879
12: -49.7084389, 2.0865293, -49.4304733, 2.1012630, -44.4658356, 44.1211090
13: -28.8353157, 21.4258156, -28.8517570, 21.3732109, -49.6324387, 49.6946411
14: -71.2095337, -6.3133373, -71.1362991, -6.3126316, -64.8969040, 64.8229599
15: -17.4426422, 24.7814636, -17.4615402, 24.7040157, -42.1466599, 42.2430038
16: -27.6229172, 23.6861038, -27.5830173, 23.7107124, -48.6255341, 48.4941711
17: -71.3113861, -3.7523537, -71.3433380, -3.7570019, -67.5543823, 67.5909882
18: -34.9166985, 11.6866770, -34.7938843, 11.6939869, -40.7825470, 40.6563568
19: -25.8128357, 5.2822590, -25.7418823, 5.2943277, -30.0240250, 29.9724503
20: -26.5838585, 4.3906598, -26.4881840, 4.4030895, -29.2443695, 29.0914192
21: -31.5383434, 10.1725540, -31.3988781, 10.1840458, -40.6308746, 40.3680496
22: -33.6918716, 7.0113111, -33.6584206, 7.0066757, -38.5606842, 38.5728912
23: -27.0221329, 8.9159298, -26.9406033, 8.9200516, -35.3585358, 35.2971077
24: -23.3182487, 9.8763866, -23.3207397, 9.8828259, -32.8819199, 32.9132462
25: -29.2831764, 6.1352167, -29.2671032, 6.1395855, -34.4545898, 34.4836807
26: -43.2354813, 7.8832808, -42.9803123, 7.8641958, -44.0619354, 43.7916183
27: -26.7183456, 11.4921379, -26.6789169, 11.5463886, -37.9197083, 37.8240623
28: -29.6672611, 7.1947894, -29.6190586, 7.2109904, -36.7168961, 36.7011108
29: -32.6820145, 9.0345192, -32.6333809, 9.0439510, -41.7259674, 41.6679001
30: -37.7093277, 7.0081739, -37.6124840, 7.0089579, -44.7182846, 44.6206589
31: -31.5460033, 7.3026309, -31.4519997, 7.3124828, -37.7113724, 37.7572403
32: -33.7941856, 6.7854567, -33.7374725, 6.9076357, -40.7018204, 40.5229301
33: -44.0177002, 16.1353188, -44.0214233, 16.0665436, -57.2502899, 57.3896866
34: -50.7628059, -4.1636643, -50.7691498, -4.1562805, -42.4145355, 42.4734573
35: -41.0175705, 7.1801343, -41.0261345, 7.1270185, -43.9787903, 44.0926208
36: -44.4991989, 5.4725399, -44.5053787, 5.5637560, -45.7064362, 45.6381989
37: -59.5348892, 2.4085946, -59.4846268, 2.4590082, -55.3538361, 55.2921219
38: -50.9750061, 8.6623163, -50.9648323, 8.7331457, -59.7081528, 59.6271477
39: -52.2280312, 14.9559460, -52.2101135, 14.9597492, -67.1877823, 67.1660614
40: -47.9362030, 8.3641653, -47.9165230, 8.4417439, -53.3632278, 53.2467194
41: -31.9493999, 15.2895422, -31.9260387, 15.4013491, -45.7495270, 45.5931778
42: -27.2789879, 10.2151470, -27.2175083, 10.2960997, -37.0852509, 36.8749771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8130232, upper bound: 14.7682100
time: 32.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8130232, upper bound: 14.8130236
time: 35.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 70.21 seconds
IS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.6593511, upper bound: 14.8130235
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.7042234, upper bound: 14.8130235
IS_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.7042235, upper bound: 14.7682098
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.7332977, upper bound: 14.8130234
IS_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.7262842, upper bound: 14.8130234
IS_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.6884390, upper bound: 14.8130234
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.7109163, upper bound: 14.7682100
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.7109163, upper bound: 14.8130236
IS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.7840025, upper bound: 14.7682100
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.7840025, upper bound: 14.8130236
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.8130232, upper bound: 14.7682100
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 70.21
Output dim: 2, lower bound: -14.8130232, upper bound: 14.8130236

## BFS IS instance: IS_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -23.4155445, 32.6699448, -23.7583504, 32.9925575, -53.9848175, 53.9907150
1: -7.3958025, 32.2304039, -7.5819073, 32.3646240, -36.2618713, 36.3220291
2: -4.4074516, 31.6145821, -4.6355391, 31.8807735, -32.9283447, 32.8730354
3: -8.6664543, 28.7502289, -8.8564892, 29.0631580, -32.3559647, 32.2297554
4: -9.6943054, 34.8812065, -9.9629297, 35.1356544, -43.0707092, 43.0885162
5: -10.8185472, 29.7111092, -11.0413885, 30.0800133, -38.0142822, 37.8554688
6: -38.5345535, 7.1772003, -38.6811714, 7.3685331, -44.0768280, 44.1479263
7: -14.9199972, 30.5036106, -15.1839457, 30.6949844, -41.4409332, 41.4873428
8: -15.0064383, 34.3224144, -15.2846375, 34.6375008, -46.7421112, 46.6912155
9: -10.0315742, 26.9787636, -10.2965069, 27.1543770, -35.4474182, 35.5308647
10: -28.0477448, 23.5866013, -28.5799026, 23.8323441, -50.0210037, 50.2985382
11: -35.4293251, 13.9594879, -35.9549942, 14.1563663, -48.9148865, 49.2535477
12: -49.1006012, 1.4386506, -49.6607056, 1.8092532, -43.5616837, 43.7730865
13: -28.6589909, 21.1212883, -28.7668381, 21.3537903, -49.3996353, 49.2688904
14: -70.5430908, -6.6857643, -71.0516739, -6.4970856, -64.0460052, 64.3659058
15: -17.1519432, 24.4994583, -17.3282394, 24.7385902, -41.8905334, 41.8276978
16: -27.1080780, 23.4503403, -27.4932213, 23.5971756, -48.0446854, 48.2159195
17: -70.9875107, -4.0909424, -71.2445526, -3.9617348, -67.0257721, 67.1536102
18: -34.6023712, 11.5705395, -34.8747940, 11.6334248, -40.3920441, 40.5791283
19: -25.5336342, 5.1562309, -25.7662945, 5.2133818, -29.7173882, 29.8602066
20: -26.3043423, 4.1994019, -26.5504837, 4.3048801, -28.8855743, 29.0328217
21: -31.1019936, 9.9230490, -31.4868908, 10.0492153, -40.0248718, 40.3179703
22: -33.4486885, 6.7575750, -33.6528244, 6.9021707, -38.2285995, 38.3064270
23: -26.7376499, 8.7334805, -26.9870739, 8.8327045, -35.0591431, 35.1557884
24: -23.0953617, 9.7754402, -23.2673836, 9.8208561, -32.6047783, 32.6984520
25: -29.0583363, 5.8823910, -29.2459145, 6.0133486, -34.1630630, 34.1707535
26: -42.7140045, 7.5356998, -43.1879921, 7.7406955, -43.4090729, 43.6963730
27: -26.4644928, 11.3625622, -26.6418762, 11.4405212, -37.5229492, 37.6222610
28: -29.4669037, 6.9863515, -29.6280098, 7.1110320, -36.4450226, 36.4347382
29: -32.4573402, 8.7712383, -32.6447144, 8.9140759, -41.3714142, 41.4159546
30: -37.4209595, 6.7250233, -37.6736221, 6.8877602, -44.3087196, 44.3986435
31: -31.1999435, 7.1840057, -31.4927444, 7.2298079, -37.4042969, 37.5418854
32: -33.4860382, 6.4741974, -33.6910210, 6.6533613, -40.1394005, 40.1652184
33: -43.6706200, 15.7730236, -43.8929749, 16.0546646, -56.9386063, 56.8612137
34: -50.5254440, -4.5153284, -50.6521645, -4.2636738, -42.1072464, 41.9928665
35: -40.7410431, 6.8114638, -40.9090271, 7.0923219, -43.6982040, 43.6009445
36: -44.2998810, 5.1729021, -44.3925171, 5.3570633, -45.2986603, 45.2140350
37: -59.2222137, 2.2229056, -59.4357033, 2.3394666, -54.9169464, 55.0102921
38: -50.6979218, 8.3949385, -50.8612137, 8.5659857, -59.2639084, 59.2561531
39: -51.9348984, 14.7846766, -52.1283684, 14.9082508, -66.8431473, 66.9130478
40: -47.6633797, 8.2186909, -47.8095131, 8.3277254, -52.9588776, 52.9887390
41: -31.6561470, 15.0442047, -31.8316555, 15.1847754, -45.2128754, 45.2789230
42: -26.9791756, 9.8638792, -27.2023811, 10.0696354, -36.4216537, 36.5555725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1627

## Relational analysis of IS_A1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6429702, upper bound: 14.8109642
time: 18.22 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6573261, upper bound: 14.8109642
time: 28.29 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -23.6293983, 32.7285843, -23.8514290, 32.9958763, -54.1971283, 54.1449890
1: -7.5351982, 32.2725677, -7.6430264, 32.3671227, -36.3940353, 36.4302063
2: -4.5403337, 31.6522007, -4.6940689, 31.8840885, -33.0508118, 32.9741859
3: -8.7665062, 28.7842522, -8.8999052, 29.0675392, -32.4379196, 32.3152847
4: -9.8722639, 34.9299011, -10.0402670, 35.1390343, -43.2389069, 43.2164154
5: -10.9337521, 29.7567406, -11.0916367, 30.0856533, -38.1176758, 37.9597397
6: -38.5784340, 7.2775793, -38.6877937, 7.4115524, -44.1707382, 44.2444763
7: -15.1065960, 30.5518837, -15.2655993, 30.6973038, -41.5968475, 41.6295242
8: -15.1852913, 34.3778992, -15.3619976, 34.6430893, -46.9152679, 46.8317108
9: -10.1715088, 27.0236645, -10.3574286, 27.1583939, -35.5710144, 35.6411400
10: -28.1530075, 23.6432610, -28.6253033, 23.8417168, -50.1270599, 50.4015121
11: -35.4635696, 13.9997854, -35.9687691, 14.1670713, -48.9599304, 49.3126831
12: -49.1529007, 1.6126189, -49.6656036, 1.8829980, -43.7032394, 43.8982925
13: -28.6969051, 21.1542187, -28.7825508, 21.3660126, -49.4501801, 49.3404846
14: -70.6826859, -6.6312771, -71.1134033, -6.4891224, -64.1935654, 64.4821243
15: -17.2265244, 24.5314980, -17.3588829, 24.7436981, -41.9702225, 41.8903809
16: -27.2656784, 23.5012169, -27.5617714, 23.6024799, -48.1896973, 48.3390961
17: -71.0370789, -4.0568314, -71.2652588, -3.9551773, -67.0819016, 67.2084274
18: -34.6273689, 11.6010342, -34.8826561, 11.6455460, -40.4190826, 40.6305542
19: -25.5678062, 5.1702237, -25.7766857, 5.2191839, -29.7675095, 29.8852577
20: -26.3299065, 4.2157555, -26.5573235, 4.3125930, -28.9511261, 29.0497437
21: -31.1463985, 9.9417067, -31.5018768, 10.0566206, -40.0888519, 40.3447189
22: -33.4974937, 6.8173261, -33.6645584, 6.9279790, -38.3053284, 38.3663254
23: -26.7668839, 8.7699318, -26.9948692, 8.8477764, -35.1154480, 35.2027588
24: -23.1205273, 9.7933283, -23.2764778, 9.8299885, -32.6430435, 32.7259521
25: -29.1004868, 5.9600697, -29.2549725, 6.0474052, -34.2414169, 34.2547836
26: -42.7588692, 7.5834379, -43.1973572, 7.7614336, -43.4756165, 43.7394562
27: -26.4956646, 11.3993387, -26.6524487, 11.4515686, -37.5697708, 37.6746902
28: -29.5041885, 7.0550199, -29.6355400, 7.1408877, -36.5140686, 36.5115891
29: -32.4981117, 8.8312607, -32.6553078, 8.9401875, -41.4383011, 41.4865685
30: -37.4438210, 6.7616615, -37.6823502, 6.9015150, -44.3453369, 44.4440117
31: -31.2454567, 7.2145672, -31.5042763, 7.2422261, -37.4924660, 37.5843430
32: -33.5288849, 6.5649738, -33.6982346, 6.6935444, -40.2224274, 40.2632065
33: -43.7475510, 15.8632832, -43.9106636, 16.0944443, -57.0569153, 56.9536057
34: -50.5762863, -4.4036245, -50.6588669, -4.2147460, -42.2210922, 42.0721436
35: -40.8002090, 6.9222326, -40.9185638, 7.1410408, -43.8190079, 43.6808472
36: -44.3555412, 5.2986903, -44.3997841, 5.4124594, -45.4183807, 45.3208694
37: -59.2902908, 2.2896309, -59.4511375, 2.3683949, -55.0247650, 55.0405197
38: -50.7727165, 8.4999828, -50.8722916, 8.6112156, -59.3839340, 59.3722763
39: -52.0129204, 14.8084660, -52.1492310, 14.9179363, -66.9308548, 66.9576950
40: -47.7229080, 8.2518740, -47.8235054, 8.3414469, -53.0302048, 53.0309372
41: -31.7051048, 15.1190434, -31.8422146, 15.2167768, -45.2935638, 45.3563843
42: -27.0155048, 9.9524927, -27.2082043, 10.1074600, -36.5143509, 36.6399269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1627

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6429702, upper bound: 14.8109642
time: 33.83 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7021990, upper bound: 14.8109642
time: 34.77 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -23.6355896, 32.7361450, -23.9210930, 33.0256310, -54.2379913, 54.2181854
1: -7.5385971, 32.2808914, -7.6918316, 32.3945312, -36.4313583, 36.4787674
2: -4.5445290, 31.6683979, -4.7843838, 31.9324799, -33.1150436, 33.0664902
3: -8.7701283, 28.7871456, -8.9462404, 29.0833683, -32.4752655, 32.3471832
4: -9.8769283, 34.9456749, -10.1293240, 35.1885300, -43.3054810, 43.3066254
5: -10.9376583, 29.7614117, -11.1357555, 30.1076450, -38.1595612, 37.9995728
6: -38.6065750, 7.2807465, -38.7751617, 7.5438919, -44.3270111, 44.3336716
7: -15.1112366, 30.5633984, -15.3228951, 30.7318363, -41.6684723, 41.6652145
8: -15.1902733, 34.3936768, -15.4410477, 34.6924362, -46.9795685, 46.9136658
9: -10.1713428, 27.0258942, -10.3838243, 27.1921139, -35.6327438, 35.6489220
10: -28.1568928, 23.6551342, -28.7027225, 23.8931198, -50.1837616, 50.5298309
11: -35.4659996, 14.0098200, -36.0152931, 14.2063532, -49.0052109, 49.3657227
12: -49.1632690, 1.6172729, -49.7036362, 1.9241457, -43.7093201, 43.9969177
13: -28.7006378, 21.1552925, -28.8232403, 21.3928337, -49.4898987, 49.3801193
14: -70.6871109, -6.6056118, -71.2524033, -6.4140396, -64.2730713, 64.6467896
15: -17.2293224, 24.5340996, -17.3912277, 24.7599373, -41.9892578, 41.9253273
16: -27.2695713, 23.5046444, -27.6166763, 23.6286983, -48.2393265, 48.3801956
17: -71.0429001, -4.0162983, -71.4552078, -3.8424530, -67.2004471, 67.4389114
18: -34.6294098, 11.6000099, -34.9251175, 11.6688042, -40.4524384, 40.6667328
19: -25.5699730, 5.1738586, -25.8096581, 5.2417698, -29.7835846, 29.9232788
20: -26.3304234, 4.2174835, -26.5673790, 4.3352089, -28.9518890, 29.0858688
21: -31.1479645, 9.9448042, -31.5314865, 10.0797129, -40.1039963, 40.3812408
22: -33.5000305, 6.8211718, -33.6991348, 6.9521351, -38.3162689, 38.4388657
23: -26.7687855, 8.7734013, -27.0207367, 8.8709660, -35.1352463, 35.2373734
24: -23.1225815, 9.7994184, -23.3324070, 9.8658457, -32.6763840, 32.7894516
25: -29.1014786, 5.9662380, -29.3032875, 6.0808992, -34.2742462, 34.3189163
26: -42.7611885, 7.5838089, -43.2225952, 7.7734528, -43.4824524, 43.7845230
27: -26.5050163, 11.4007339, -26.6979332, 11.5150385, -37.6649246, 37.7229919
28: -29.5054073, 7.0586138, -29.6535549, 7.1818824, -36.5607224, 36.5341568
29: -32.4997215, 8.8352280, -32.6858635, 8.9639511, -41.4636726, 41.5210915
30: -37.4443016, 6.7650595, -37.6962738, 6.9294844, -44.3737869, 44.4613342
31: -31.2486916, 7.2252760, -31.5421219, 7.2819152, -37.5115471, 37.6416321
32: -33.5565948, 6.5670033, -33.7881012, 6.8291483, -40.3857422, 40.3551025
33: -43.7510300, 15.8656244, -43.9338722, 16.1372070, -57.1004333, 57.0034180
34: -50.5937576, -4.4009571, -50.7203102, -4.1259656, -42.2932205, 42.1788330
35: -40.8094902, 6.9247980, -40.9552765, 7.1868143, -43.8403931, 43.7590408
36: -44.3837738, 5.3015919, -44.4871597, 5.5332842, -45.5391846, 45.4386749
37: -59.3022461, 2.2924509, -59.4972420, 2.4503098, -55.0970993, 55.1225662
38: -50.7941666, 8.5024185, -50.9528580, 8.7151241, -59.5092926, 59.4552765
39: -52.0217590, 14.8101168, -52.1923752, 14.9653740, -66.9871368, 67.0024948
40: -47.7455254, 8.2537680, -47.8950806, 8.4592342, -53.1671448, 53.1061096
41: -31.7308655, 15.1219730, -31.9277306, 15.3481960, -45.4469528, 45.4477539
42: -27.0334778, 9.9550552, -27.2704468, 10.2066269, -36.6194763, 36.7133026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1627

## Relational analysis of IS_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6878400, upper bound: 14.8109641
time: 20.74 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7312643, upper bound: 14.8109641
time: 36.59 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -23.6500301, 32.7484512, -23.8510113, 33.0231361, -54.2232666, 54.1608658
1: -7.5286760, 32.2786293, -7.6278267, 32.3952293, -36.4037781, 36.4122581
2: -4.6103506, 31.6877708, -4.6988277, 31.9269028, -33.1583862, 33.0071564
3: -8.7891998, 28.8092690, -8.9008465, 29.0798340, -32.4832840, 32.3299866
4: -9.9086676, 34.9657555, -10.0296268, 35.1824684, -43.3245087, 43.2400055
5: -10.9438524, 29.7841530, -11.0877075, 30.1049595, -38.1611328, 37.9717636
6: -38.6292305, 7.4558954, -38.7589455, 7.4465904, -44.2501373, 44.4979019
7: -15.0761957, 30.5648022, -15.2382908, 30.7285748, -41.6055756, 41.6022415
8: -15.2375116, 34.4270782, -15.3651142, 34.6862717, -47.0002823, 46.8739471
9: -10.1372871, 27.0456047, -10.3419476, 27.1663704, -35.5823517, 35.6667328
10: -28.1964760, 23.6575680, -28.6225319, 23.8837204, -50.2789459, 50.4631195
11: -35.5313797, 14.0744305, -35.9738197, 14.2268972, -49.1140137, 49.3961182
12: -49.1760674, 1.5949473, -49.6950417, 1.8711009, -43.7046280, 43.9462738
13: -28.7761784, 21.1977596, -28.8142490, 21.3747520, -49.5557861, 49.4186096
14: -70.8397293, -6.5908947, -71.1392899, -6.4180145, -64.4217148, 64.5483932
15: -17.2620869, 24.5442448, -17.3707428, 24.7570343, -42.0191193, 41.9149857
16: -27.2432365, 23.4991417, -27.5371590, 23.6155300, -48.2028351, 48.3196030
17: -71.2173004, -4.0120449, -71.2804871, -3.8467751, -67.3705292, 67.2684402
18: -34.6795654, 11.6424656, -34.8931122, 11.6691284, -40.5115280, 40.6808701
19: -25.6116695, 5.2197013, -25.7858295, 5.2497778, -29.8331451, 29.9468193
20: -26.3513298, 4.2869878, -26.5623093, 4.3429198, -28.9734344, 29.1193733
21: -31.1891499, 10.0292377, -31.5070744, 10.1066465, -40.1751099, 40.4450912
22: -33.5422058, 6.8735299, -33.6717720, 6.9610977, -38.4052734, 38.4105682
23: -26.8096352, 8.8097610, -27.0042667, 8.8728943, -35.1791382, 35.2546539
24: -23.1983948, 9.8451767, -23.2860336, 9.8647079, -32.7581482, 32.7845001
25: -29.1565895, 5.9833674, -29.2607994, 6.0719600, -34.3269272, 34.2866554
26: -42.8039551, 7.6430798, -43.2073441, 7.7918801, -43.5616150, 43.8050232
27: -26.5467873, 11.4942341, -26.6822891, 11.4783983, -37.6607513, 37.8159332
28: -29.5360069, 7.1050844, -29.6471672, 7.1581130, -36.5614777, 36.5785904
29: -32.5404892, 8.9009066, -32.6630173, 8.9767952, -41.5172844, 41.5639229
30: -37.4950333, 6.8645487, -37.6852798, 6.9571934, -44.4522247, 44.5498276
31: -31.2807655, 7.2393293, -31.5149555, 7.2737064, -37.5384445, 37.6204681
32: -33.5781212, 6.6958485, -33.7755585, 6.6996202, -40.2777405, 40.4714050
33: -43.7634201, 15.8609295, -43.9296722, 16.0817642, -57.0745850, 57.0098495
34: -50.6123009, -4.3448610, -50.7152824, -4.2186279, -42.2468719, 42.2179871
35: -40.8211746, 6.9143858, -40.9498634, 7.1227789, -43.8130875, 43.7517014
36: -44.3929710, 5.3798108, -44.4756622, 5.4045238, -45.4412079, 45.4879303
37: -59.3225861, 2.3425813, -59.4924355, 2.3662515, -55.0466461, 55.1908493
38: -50.7923279, 8.5485067, -50.9332504, 8.5957098, -59.3880386, 59.4817581
39: -52.0405769, 14.8453760, -52.1817017, 14.9206219, -66.9611969, 67.0270767
40: -47.7451820, 8.3681622, -47.8872108, 8.3447371, -53.0583267, 53.2426987
41: -31.7562790, 15.2682781, -31.9130154, 15.2361488, -45.3663025, 45.5815048
42: -27.0522919, 10.0489578, -27.2596054, 10.1174049, -36.5422363, 36.7957764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=176, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1627

## Relational analysis of IS_A1_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6720689, upper bound: 14.8109641
time: 37.70 seconds

## Relational analysis of IS_A1_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7242453, upper bound: 14.8109641
time: 30.55 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -23.8640099, 32.8071327, -23.9441090, 33.0264359, -54.4357147, 54.3152161
1: -7.6680899, 32.3208084, -7.6889553, 32.3977623, -36.5359573, 36.5204391
2: -4.7432394, 31.7253971, -4.7573528, 31.9302101, -33.2808762, 33.1082993
3: -8.8893471, 28.8432426, -8.9442673, 29.0841599, -32.5653000, 32.4154816
4: -10.0866537, 35.0144730, -10.1069860, 35.1858292, -43.4928436, 43.3679581
5: -11.0590782, 29.8297291, -11.1379557, 30.1105957, -38.2645493, 38.0759964
6: -38.6730652, 7.5564499, -38.7655945, 7.4895811, -44.3440628, 44.5945816
7: -15.2627630, 30.6130867, -15.3199329, 30.7308750, -41.7614746, 41.7444305
8: -15.4163561, 34.4826012, -15.4425182, 34.6918755, -47.1734009, 47.0145187
9: -10.2772837, 27.0904922, -10.4028683, 27.1703815, -35.7060089, 35.7770462
10: -28.3017921, 23.7142124, -28.6679153, 23.8931293, -50.3851013, 50.5660248
11: -35.5656395, 14.1146927, -35.9875450, 14.2376270, -49.1591110, 49.4552841
12: -49.2283707, 1.7690611, -49.6999092, 1.9448509, -43.8462601, 44.0717697
13: -28.8142548, 21.2306690, -28.8299789, 21.3870430, -49.6064606, 49.4902344
14: -70.9794006, -6.5365047, -71.2010345, -6.4099960, -64.5694046, 64.6645279
15: -17.3367596, 24.5763454, -17.4014225, 24.7621517, -42.0989113, 41.9777679
16: -27.4008923, 23.5500317, -27.6056824, 23.6208458, -48.3479614, 48.4427872
17: -71.2669144, -3.9779568, -71.3011322, -3.8402462, -67.4266663, 67.3231735
18: -34.7045822, 11.6728134, -34.9009628, 11.6813145, -40.5385971, 40.7322769
19: -25.6458435, 5.2337065, -25.7962112, 5.2555761, -29.8833466, 29.9718475
20: -26.3769321, 4.3033338, -26.5691452, 4.3506193, -29.0390015, 29.1362991
21: -31.2335625, 10.0477562, -31.5220490, 10.1140461, -40.2390900, 40.4717941
22: -33.5911179, 6.9332643, -33.6835098, 6.9869547, -38.4820709, 38.4704514
23: -26.8389149, 8.8461838, -27.0120544, 8.8879547, -35.2354050, 35.3016357
24: -23.2236004, 9.8630095, -23.2951469, 9.8738365, -32.7964478, 32.8119659
25: -29.1987839, 6.0610356, -29.2698822, 6.1060152, -34.4053421, 34.3706627
26: -42.8487968, 7.6908131, -43.2166824, 7.8126307, -43.6281967, 43.8481522
27: -26.5779018, 11.5309925, -26.6928902, 11.4894218, -37.7075195, 37.8683205
28: -29.5732422, 7.1737757, -29.6547031, 7.1879997, -36.6305313, 36.6555023
29: -32.5812531, 8.9609985, -32.6736298, 9.0029345, -41.5841866, 41.6346283
30: -37.5179482, 6.9011765, -37.6940117, 6.9709558, -44.4889030, 44.5951881
31: -31.3262577, 7.2699022, -31.5264606, 7.2861433, -37.6265945, 37.6629333
32: -33.6209106, 6.7868328, -33.7828026, 6.7398043, -40.3607140, 40.5696335
33: -43.8402748, 15.9512606, -43.9472847, 16.1216068, -57.1927643, 57.1024399
34: -50.6630478, -4.2331448, -50.7220001, -4.1697254, -42.3606262, 42.2972870
35: -40.8803444, 7.0251718, -40.9593353, 7.1714611, -43.9337769, 43.8315811
36: -44.4485588, 5.5056725, -44.4829483, 5.4598894, -45.5608215, 45.5947952
37: -59.3906708, 2.4093189, -59.5079422, 2.3951497, -55.1543732, 55.2211838
38: -50.8670540, 8.6536417, -50.9443245, 8.6409721, -59.5080261, 59.5979652
39: -52.1185303, 14.8691959, -52.2025070, 14.9303455, -67.0488739, 67.0717010
40: -47.8047562, 8.4012680, -47.9012070, 8.3584471, -53.1297531, 53.2848511
41: -31.8051796, 15.3431540, -31.9235535, 15.2681589, -45.4468079, 45.6590729
42: -27.0885963, 10.1376019, -27.2654133, 10.1552248, -36.6348801, 36.8801575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=176, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1627

## Relational analysis of IS_A1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6720689, upper bound: 14.8109641
time: 32.12 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7690824, upper bound: 14.8109641
time: 14.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -23.6832714, 32.8237000, -23.8422832, 32.8686028, -54.0874481, 54.2018890
1: -7.5349503, 32.2881699, -7.6317711, 32.3128052, -36.3623123, 36.3854713
2: -4.6082377, 31.7449627, -4.7189035, 31.7565994, -32.9398346, 33.0748596
3: -8.8096848, 28.8775425, -8.9142876, 28.9056911, -32.2835541, 32.3806267
4: -9.9323063, 35.0178795, -10.0636215, 35.0424919, -43.1763000, 43.2967987
5: -11.0005951, 29.8810844, -11.1173239, 29.9174347, -37.9648438, 38.0924225
6: -38.6627731, 7.3678131, -38.6719666, 7.4417973, -44.3738251, 44.3109360
7: -15.0863647, 30.5765133, -15.2278566, 30.6095028, -41.4957886, 41.5820007
8: -15.2693176, 34.5076332, -15.4071846, 34.5380554, -46.8387451, 46.9722595
9: -10.2029343, 27.1451054, -10.2793531, 27.2041912, -35.6839142, 35.6131897
10: -28.3520851, 23.8695030, -28.4106216, 23.9571953, -50.4952393, 50.2648468
11: -35.7508888, 14.2376451, -35.7806244, 14.2995510, -49.3899689, 49.2991180
12: -49.3519020, 1.7786703, -49.3884697, 1.9713349, -43.9555817, 43.7783890
13: -28.7698479, 21.2952919, -28.7982330, 21.3317833, -49.5271759, 49.4920425
14: -70.8536301, -6.4705620, -70.9339294, -6.4054508, -64.4481812, 64.4633636
15: -17.3412476, 24.6509838, -17.4040890, 24.6819916, -42.0232391, 42.0550728
16: -27.3600407, 23.6191635, -27.4568424, 23.6824627, -48.3591614, 48.2971878
17: -71.1052322, -3.9124870, -71.1326447, -3.8866310, -67.2185974, 67.2201538
18: -34.7237091, 11.6255989, -34.7431679, 11.6580391, -40.5441208, 40.5218163
19: -25.6690407, 5.2435284, -25.6974525, 5.2631702, -29.8639374, 29.8989563
20: -26.4469795, 4.3407965, -26.4702721, 4.3699331, -29.0477371, 29.0226250
21: -31.3169880, 10.1079197, -31.3533287, 10.1484823, -40.3587189, 40.2661972
22: -33.5647202, 6.9021187, -33.6128311, 6.9579835, -38.3759308, 38.3953781
23: -26.8816051, 8.8507004, -26.9060421, 8.8790112, -35.1803894, 35.1900024
24: -23.2302971, 9.8370571, -23.2545910, 9.8397675, -32.7568703, 32.8066902
25: -29.1709099, 6.0173926, -29.2094040, 6.0685310, -34.2781677, 34.3004913
26: -42.9048920, 7.7195296, -42.9460907, 7.8196998, -43.6769943, 43.5971298
27: -26.6020279, 11.4361534, -26.6213284, 11.4722691, -37.7061844, 37.7006416
28: -29.5621796, 7.0959892, -29.5925121, 7.1390691, -36.5441132, 36.5601883
29: -32.5625076, 8.9324312, -32.5926743, 8.9909668, -41.5534744, 41.5251045
30: -37.5673676, 6.9270239, -37.5891991, 6.9649706, -44.5323372, 44.5162239
31: -31.3622017, 7.2452283, -31.4008980, 7.2583838, -37.4997253, 37.6445923
32: -33.6431885, 6.6632557, -33.6402092, 6.7310781, -40.3742676, 40.3034668
33: -43.8829536, 15.9290857, -43.9753914, 15.9829578, -57.0160522, 57.1146011
34: -50.6621475, -4.3609734, -50.6983643, -4.2960925, -42.1540527, 42.2011108
35: -40.9114838, 6.9653702, -40.9762650, 7.0300255, -43.7564697, 43.8306808
36: -44.3982391, 5.3047290, -44.4107742, 5.3866968, -45.4156799, 45.3695984
37: -59.3787804, 2.3068924, -59.4222069, 2.3508158, -55.0694885, 55.1262360
38: -50.8324890, 8.5125132, -50.8730545, 8.5850439, -59.4175339, 59.3855667
39: -52.0895767, 14.8793678, -52.1442375, 14.9055700, -66.9951477, 67.0236053
40: -47.8012733, 8.2896061, -47.8298111, 8.3111343, -53.0957794, 53.0852966
41: -31.8194580, 15.1884575, -31.8290443, 15.2385769, -45.4383240, 45.4028931
42: -27.1392670, 10.0929079, -27.1492348, 10.1626759, -36.7437897, 36.6896057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1627

## Relational analysis of IS_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6640504, upper bound: 14.7966451
time: 17.22 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6640504, upper bound: 14.8109643
time: 39.31 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -23.8972111, 32.8823776, -23.9353790, 32.8719521, -54.2998581, 54.3562851
1: -7.6743460, 32.3303680, -7.6928797, 32.3153038, -36.4944916, 36.4936752
2: -4.7411251, 31.7825947, -4.7774477, 31.7598648, -33.0622406, 33.1759605
3: -8.9097795, 28.9115391, -8.9577665, 28.9100838, -32.3654938, 32.4662552
4: -10.1101875, 35.0665054, -10.1409645, 35.0458336, -43.3445282, 43.4247665
5: -11.1158705, 29.9267635, -11.1676121, 29.9230881, -38.0682220, 38.1967773
6: -38.7067108, 7.4683561, -38.6785965, 7.4848576, -44.4678040, 44.4075394
7: -15.2728996, 30.6248055, -15.3095284, 30.6118088, -41.6516876, 41.7242203
8: -15.4481335, 34.5631256, -15.4845457, 34.5436478, -47.0117340, 47.1128159
9: -10.3429317, 27.1900444, -10.3402624, 27.2082062, -35.8075714, 35.7235260
10: -28.4573860, 23.9261646, -28.4560795, 23.9665489, -50.6013107, 50.3678360
11: -35.7851486, 14.2779970, -35.7944031, 14.3101959, -49.4351349, 49.3583145
12: -49.4042282, 1.9525642, -49.3933640, 2.0451159, -44.0973129, 43.9036179
13: -28.8079758, 21.3283195, -28.8139553, 21.3440361, -49.5777893, 49.5636368
14: -70.9933167, -6.4160938, -70.9957275, -6.3975086, -64.5958099, 64.5796356
15: -17.4158096, 24.6830502, -17.4347496, 24.6870899, -42.1028976, 42.1177979
16: -27.5176601, 23.6700859, -27.5254555, 23.6877193, -48.5042496, 48.4204102
17: -71.1548386, -3.8782883, -71.1532898, -3.8799992, -67.2748413, 67.2750015
18: -34.7488403, 11.6560459, -34.7510452, 11.6701813, -40.5712814, 40.5732117
19: -25.7031803, 5.2575760, -25.7078667, 5.2689743, -29.9141998, 29.9240074
20: -26.4725990, 4.3571301, -26.4770851, 4.3776102, -29.1133652, 29.0395279
21: -31.3614388, 10.1265707, -31.3683052, 10.1558609, -40.4227753, 40.2929611
22: -33.6135597, 6.9618554, -33.6245499, 6.9838047, -38.4526291, 38.4552536
23: -26.9108925, 8.8871193, -26.9138298, 8.8941059, -35.2367477, 35.2370071
24: -23.2554932, 9.8549070, -23.2636890, 9.8488731, -32.7951965, 32.8341408
25: -29.2130928, 6.0950913, -29.2184906, 6.1026020, -34.3566132, 34.3844757
26: -42.9497910, 7.7672834, -42.9555168, 7.8404093, -43.7437973, 43.6402054
27: -26.6332150, 11.4729319, -26.6319427, 11.4832802, -37.7530823, 37.7529449
28: -29.5995121, 7.1646552, -29.6000538, 7.1689434, -36.6132736, 36.6369781
29: -32.6032562, 8.9924936, -32.6032791, 9.0171022, -41.6203575, 41.5957718
30: -37.5902977, 6.9637289, -37.5979424, 6.9786415, -44.5689392, 44.5616722
31: -31.4077873, 7.2758732, -31.4124641, 7.2708349, -37.5879669, 37.6870575
32: -33.6860161, 6.7541246, -33.6474533, 6.7712736, -40.4572906, 40.4015770
33: -43.9598389, 16.0192890, -43.9929924, 16.0228176, -57.1344147, 57.2070618
34: -50.7129669, -4.2492661, -50.7050552, -4.2472029, -42.2679520, 42.2803497
35: -40.9707489, 7.0761113, -40.9857559, 7.0786791, -43.8771973, 43.9105148
36: -44.4538498, 5.4305568, -44.4180756, 5.4421310, -45.5353699, 45.4764633
37: -59.4468765, 2.3735728, -59.4376335, 2.3797255, -55.1773529, 55.1564789
38: -50.9073639, 8.6175661, -50.8842087, 8.6303043, -59.5376663, 59.5017738
39: -52.1676064, 14.9031496, -52.1650467, 14.9152870, -67.0828934, 67.0681992
40: -47.8608627, 8.3226929, -47.8437843, 8.3248911, -53.1672897, 53.1275024
41: -31.8684063, 15.2633705, -31.8395710, 15.2705746, -45.5189514, 45.4804382
42: -27.1755009, 10.1814671, -27.1550674, 10.2004967, -36.8364716, 36.7739258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1627

## Relational analysis of IS_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6640504, upper bound: 14.7966451
time: 47.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6640504, upper bound: 14.8109643
time: 36.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -23.9033833, 32.8899918, -24.0049477, 32.9016609, -54.3407516, 54.4295197
1: -7.6777506, 32.3386841, -7.7416325, 32.3426857, -36.5317535, 36.5421753
2: -4.7453318, 31.7987862, -4.8677502, 31.8082848, -33.1265640, 33.2682495
3: -8.9133778, 28.9144554, -9.0041142, 28.9259491, -32.4028320, 32.4982033
4: -10.1148090, 35.0822716, -10.2300196, 35.0954170, -43.4110870, 43.5149765
5: -11.1197224, 29.9314270, -11.2117519, 29.9450932, -38.1101151, 38.2366257
6: -38.7348289, 7.4715014, -38.7658806, 7.6171703, -44.6240768, 44.4967117
7: -15.2775927, 30.6363411, -15.3668118, 30.6463928, -41.7233124, 41.7599030
8: -15.4531031, 34.5788879, -15.5636044, 34.5930710, -47.0761337, 47.1948166
9: -10.3427725, 27.1922817, -10.3666668, 27.2419205, -35.8692932, 35.7312889
10: -28.4613113, 23.9380512, -28.5334969, 24.0179138, -50.6579971, 50.4960785
11: -35.7876129, 14.2880020, -35.8409882, 14.3494530, -49.4803314, 49.4113922
12: -49.4145355, 1.9572806, -49.4313965, 2.0861416, -44.1032944, 44.0022278
13: -28.8116722, 21.3294125, -28.8547096, 21.3709106, -49.6175919, 49.6033173
14: -70.9976654, -6.3904285, -71.1345749, -6.3224907, -64.6751709, 64.7441483
15: -17.4186077, 24.6856441, -17.4671135, 24.7033176, -42.1219254, 42.1527557
16: -27.5215874, 23.6734524, -27.5802937, 23.7140675, -48.5538712, 48.4614639
17: -71.1606674, -3.8377552, -71.3431854, -3.7673378, -67.3933258, 67.5054321
18: -34.7508621, 11.6550455, -34.7935257, 11.6933498, -40.6046906, 40.6092186
19: -25.7053909, 5.2612123, -25.7407322, 5.2915525, -29.9302368, 29.9619865
20: -26.4730892, 4.3588758, -26.4871941, 4.4002166, -29.1141281, 29.0756607
21: -31.3630199, 10.1296196, -31.3979206, 10.1789217, -40.4378738, 40.3295746
22: -33.6160927, 6.9656796, -33.6590500, 7.0078621, -38.4635925, 38.5277252
23: -26.9128036, 8.8906021, -26.9397526, 8.9172249, -35.2565765, 35.2716141
24: -23.2575874, 9.8609829, -23.3195343, 9.8846016, -32.8284149, 32.8975868
25: -29.2141037, 6.1012616, -29.2667427, 6.1360188, -34.3893051, 34.4484978
26: -42.9520950, 7.7676535, -42.9807510, 7.8525395, -43.7505646, 43.6852875
27: -26.6425591, 11.4743433, -26.6774864, 11.5467548, -37.8482437, 37.8013649
28: -29.6007462, 7.1682463, -29.6181297, 7.2099795, -36.6599655, 36.6596680
29: -32.6049385, 8.9964495, -32.6338577, 9.0408707, -41.6458092, 41.6303062
30: -37.5907860, 6.9671373, -37.6118622, 7.0066051, -44.5973892, 44.5789986
31: -31.4110088, 7.2865543, -31.4502754, 7.3104496, -37.6070023, 37.7441940
32: -33.7137070, 6.7561560, -33.7372551, 6.9068031, -40.6205101, 40.4934120
33: -43.9632797, 16.0216560, -44.0163078, 16.0655746, -57.1780853, 57.2569122
34: -50.7303696, -4.2466288, -50.7664490, -4.1584072, -42.3400116, 42.3870392
35: -40.9800034, 7.0786943, -41.0225296, 7.1245604, -43.8985672, 43.9887390
36: -44.4821053, 5.4334226, -44.5054092, 5.5628490, -45.6561737, 45.5942230
37: -59.4587936, 2.3763914, -59.4836617, 2.4616795, -55.2494888, 55.2384720
38: -50.9288216, 8.6199684, -50.9645309, 8.7341356, -59.6629562, 59.5844994
39: -52.1764526, 14.9048252, -52.2083435, 14.9626789, -67.1391296, 67.1131668
40: -47.8834877, 8.3246517, -47.9152870, 8.4426870, -53.3042603, 53.2026138
41: -31.8941765, 15.2662525, -31.9250984, 15.4020138, -45.6723175, 45.5718002
42: -27.1934147, 10.1840124, -27.2173119, 10.2996731, -36.9415436, 36.8472710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1627

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7198602, upper bound: 14.8109643
time: 47.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7632553, upper bound: 14.8109643
time: 30.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -23.9890614, 33.0343018, -23.9347935, 32.8720169, -54.3971252, 54.5128098
1: -7.7106504, 32.3909187, -7.6901217, 32.3147583, -36.5483856, 36.5414276
2: -4.8077888, 31.9158688, -4.7815981, 31.7591438, -33.1408997, 33.3062820
3: -8.9855881, 29.0908012, -8.9637814, 28.9090061, -32.4528046, 32.6303864
4: -10.1589890, 35.1752625, -10.1411343, 35.0450058, -43.4086304, 43.5248642
5: -11.2010517, 30.1198864, -11.1749039, 29.9224186, -38.1676025, 38.3811798
6: -38.7622261, 7.4889469, -38.6780128, 7.4797955, -44.5619888, 44.4307480
7: -15.3505297, 30.7258186, -15.3088503, 30.6102409, -41.7588196, 41.7983475
8: -15.5035772, 34.6854248, -15.4841852, 34.5428772, -47.0805206, 47.2373352
9: -10.4084167, 27.2277107, -10.3369551, 27.2065506, -35.8939667, 35.7382812
10: -28.6742039, 24.0122414, -28.4543858, 23.9730835, -50.8321838, 50.4484787
11: -36.0100517, 14.3311863, -35.7931442, 14.3146591, -49.6798782, 49.4099808
12: -49.6984024, 2.0840735, -49.3915215, 2.0539284, -44.3494110, 44.0759659
13: -28.8321648, 21.4251900, -28.8094101, 21.3447952, -49.6135101, 49.6392670
14: -71.2069550, -6.3387909, -70.9922943, -6.3884144, -64.8185425, 64.6535034
15: -17.4408855, 24.7791977, -17.4260921, 24.6868038, -42.1276894, 42.2052917
16: -27.6211567, 23.6828671, -27.5220680, 23.6838970, -48.5957642, 48.4104385
17: -71.3062820, -3.7925205, -71.1512985, -3.8706417, -67.4356384, 67.3587799
18: -34.9149857, 11.6880875, -34.7504044, 11.6696606, -40.7534027, 40.6159134
19: -25.8111134, 5.2790108, -25.7076054, 5.2706037, -29.9991760, 29.9472427
20: -26.5836143, 4.3893213, -26.4772701, 4.3794169, -29.2200699, 29.0949287
21: -31.5373459, 10.1697426, -31.3675823, 10.1602468, -40.6054230, 40.3474503
22: -33.6897888, 7.0083356, -33.6226425, 6.9802599, -38.5189209, 38.5143356
23: -27.0207405, 8.9133730, -26.9132729, 8.8943567, -35.3300095, 35.2735786
24: -23.3168945, 9.8706379, -23.2628460, 9.8461666, -32.8448372, 32.8544922
25: -29.2826786, 6.1301174, -29.2173862, 6.1031089, -34.4124680, 34.4227295
26: -43.2337036, 7.8837056, -42.9538383, 7.8497562, -44.0262222, 43.7594299
27: -26.7094288, 11.4911070, -26.6320095, 11.4818048, -37.8248138, 37.7698898
28: -29.6664448, 7.1923528, -29.5998859, 7.1666961, -36.6663208, 36.6801071
29: -32.6807938, 9.0314379, -32.6014404, 9.0178585, -41.6986542, 41.6328773
30: -37.7092857, 7.0053425, -37.5972672, 6.9795742, -44.6888580, 44.6026077
31: -31.5434418, 7.2925463, -31.4122772, 7.2711091, -37.6723709, 37.7330856
32: -33.7668724, 6.7843876, -33.6466637, 6.7692356, -40.5361099, 40.4310532
33: -44.0149574, 16.1337376, -43.9961853, 16.0215397, -57.1712875, 57.3554916
34: -50.7457542, -4.1648731, -50.7067299, -4.2491770, -42.2534790, 42.4099579
35: -41.0087967, 7.1790290, -40.9879189, 7.0771103, -43.8716431, 44.0554352
36: -44.4712906, 5.4712725, -44.4169769, 5.4382715, -45.5244446, 45.5494003
37: -59.5236969, 2.4067116, -59.4364014, 2.3744345, -55.2142487, 55.2402573
38: -50.9540710, 8.6607037, -50.8830605, 8.6270113, -59.5810814, 59.5437622
39: -52.2200623, 14.9548292, -52.1643677, 14.9106865, -67.1307526, 67.1191940
40: -47.9140739, 8.3626947, -47.8436699, 8.3225918, -53.2176666, 53.1735382
41: -31.9241943, 15.2879534, -31.8390369, 15.2657986, -45.5826340, 45.5065613
42: -27.2612820, 10.2134790, -27.1546364, 10.1944466, -36.9647064, 36.8243103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1627

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7819378, upper bound: 14.7966450
time: 38.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7819378, upper bound: 14.8109642
time: 35.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -23.8981419, 33.0382080, -23.7943935, 32.8433762, -54.2761307, 54.3810501
1: -7.6505108, 32.3965073, -7.6019087, 32.3002701, -36.4651794, 36.4701271
2: -4.7511649, 31.9284630, -4.7413368, 31.7702465, -33.0879059, 33.2922592
3: -8.9440422, 29.0888081, -8.9116726, 28.8913670, -32.3830872, 32.6017609
4: -10.0833015, 35.1872330, -10.0552835, 35.0462952, -43.3312912, 43.4629364
5: -11.1526508, 30.1184216, -11.1057873, 29.8992081, -38.0855408, 38.3371201
6: -38.7831764, 7.4473248, -38.7219315, 7.5133276, -44.6264954, 44.4210815
7: -15.2704124, 30.7347107, -15.1826477, 30.5967846, -41.6520233, 41.7143936
8: -15.4281654, 34.6950073, -15.3874454, 34.5373154, -46.9887619, 47.1618652
9: -10.3449221, 27.2255611, -10.2258396, 27.1955948, -35.8176651, 35.6501312
10: -28.6308613, 24.0142612, -28.4282913, 23.9682350, -50.7827148, 50.4737396
11: -35.9977112, 14.3300533, -35.8064880, 14.3141241, -49.6628647, 49.4208374
12: -49.7034531, 2.0121651, -49.3776665, 1.9238496, -44.2847748, 43.9783783
13: -28.8194160, 21.4133530, -28.8127575, 21.3393707, -49.5694580, 49.6404495
14: -71.1472702, -6.3213863, -70.9937744, -6.3675327, -64.7797394, 64.6723862
15: -17.4116631, 24.7762337, -17.3853035, 24.6713829, -42.0830460, 42.1615372
16: -27.5537281, 23.6806602, -27.4220543, 23.6594582, -48.5012970, 48.3272552
17: -71.2905426, -3.7590256, -71.2925873, -3.7917366, -67.4988098, 67.5335617
18: -34.9087143, 11.6743956, -34.7682343, 11.6629820, -40.7325287, 40.6276932
19: -25.8022842, 5.2763500, -25.7068825, 5.2796721, -29.9973526, 29.9277878
20: -26.5769272, 4.3828502, -26.4620934, 4.3861699, -29.2249908, 29.0443840
21: -31.5231762, 10.1650686, -31.3534679, 10.1650581, -40.6026154, 40.3112259
22: -33.6799698, 6.9852791, -33.6088028, 6.9457159, -38.4849472, 38.4950714
23: -27.0142021, 8.9006214, -26.9105301, 8.8822203, -35.3094254, 35.2451935
24: -23.3089123, 9.8671913, -23.2943878, 9.8644962, -32.8534660, 32.8767128
25: -29.2739620, 6.1008620, -29.2240486, 6.0602431, -34.3653488, 34.4041672
26: -43.2259903, 7.8622751, -42.9346619, 7.8151135, -44.0038147, 43.7238007
27: -26.7075634, 11.4809904, -26.6470013, 11.5090084, -37.8656311, 37.7734337
28: -29.6595898, 7.1646228, -29.5811176, 7.1404924, -36.6380310, 36.6318207
29: -32.6712532, 9.0082121, -32.5918427, 8.9825802, -41.6538315, 41.6000557
30: -37.7004089, 6.9943485, -37.5888824, 6.9715347, -44.6719437, 44.5832291
31: -31.5342712, 7.2900357, -31.4053421, 7.2809443, -37.6665878, 37.6843262
32: -33.7868729, 6.7449598, -33.6941109, 6.8151779, -40.6020508, 40.4390717
33: -43.9998741, 16.0953217, -43.9434166, 15.9750576, -57.1393509, 57.2701187
34: -50.7559967, -4.2129440, -50.7177734, -4.2702589, -42.2903137, 42.3587494
35: -41.0078964, 7.1310649, -40.9661179, 7.0140676, -43.8554153, 43.9706345
36: -44.4917679, 5.4166679, -44.4491806, 5.4353261, -45.5684204, 45.5174408
37: -59.5192032, 2.3794341, -59.4152222, 2.3908191, -55.2886658, 55.1819916
38: -50.9637375, 8.6168222, -50.8892059, 8.6268330, -59.5905685, 59.5060272
39: -52.2069626, 14.9460793, -52.1306534, 14.9350786, -67.1420441, 67.0767365
40: -47.9220657, 8.3502998, -47.8561974, 8.4079208, -53.3159027, 53.1735382
41: -31.9387283, 15.2569914, -31.8762856, 15.3241444, -45.6645508, 45.5112457
42: -27.2731094, 10.1770573, -27.1809444, 10.2061510, -36.9981689, 36.7923508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=223, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1627

## Relational analysis of IS_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109638, upper bound: 14.7518362
time: 38.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109638, upper bound: 14.7661503
time: 42.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -23.9912205, 33.0415344, -24.0083733, 32.9021225, -54.4304886, 54.5935593
1: -7.7116232, 32.3990135, -7.7413044, 32.3424301, -36.5734024, 36.6022682
2: -4.8096628, 31.9317207, -4.8742018, 31.8078327, -33.1889801, 33.4146957
3: -8.9874706, 29.0931129, -9.0118484, 28.9253674, -32.4686203, 32.6838531
4: -10.1606245, 35.1906052, -10.2331657, 35.0949249, -43.4591522, 43.6311340
5: -11.2029171, 30.1240559, -11.2210388, 29.9449100, -38.1899033, 38.4406204
6: -38.7897987, 7.4903250, -38.7658463, 7.6138744, -44.7231979, 44.5149841
7: -15.3520889, 30.7369995, -15.3692188, 30.6451530, -41.7942200, 41.8703613
8: -15.5055399, 34.7006340, -15.5662527, 34.5928383, -47.1292419, 47.3349457
9: -10.4058590, 27.2296162, -10.3657961, 27.2406006, -35.9280243, 35.7737732
10: -28.6762810, 24.0236111, -28.5335846, 24.0249310, -50.8857498, 50.5798416
11: -36.0114517, 14.3407249, -35.8408012, 14.3544464, -49.7220306, 49.4661255
12: -49.7083359, 2.0859261, -49.4300003, 2.0978198, -44.4100647, 44.1199570
13: -28.8351517, 21.4256439, -28.8508854, 21.3723660, -49.6411285, 49.6911087
14: -71.2089996, -6.3134212, -71.1334457, -6.3129978, -64.8960037, 64.8200226
15: -17.4423370, 24.7813606, -17.4598503, 24.7034531, -42.1457901, 42.2412109
16: -27.6223011, 23.6859779, -27.5796623, 23.7103920, -48.6245117, 48.4723511
17: -71.3111496, -3.7524796, -71.3421783, -3.7574921, -67.5536575, 67.5896988
18: -34.9165993, 11.6865578, -34.7933350, 11.6933842, -40.7839470, 40.6548462
19: -25.8126793, 5.2821403, -25.7411251, 5.2936907, -30.0224609, 29.9780693
20: -26.5837650, 4.3905478, -26.4877110, 4.4024944, -29.2419243, 29.1099930
21: -31.5381622, 10.1724548, -31.3979607, 10.1836567, -40.6293488, 40.3752594
22: -33.6917725, 7.0110860, -33.6577225, 7.0054083, -38.5448151, 38.5719452
23: -27.0220127, 8.9156532, -26.9398193, 8.9186459, -35.3564224, 35.3015594
24: -23.3180294, 9.8762932, -23.3196011, 9.8823109, -32.8809013, 32.9151230
25: -29.2830505, 6.1349082, -29.2663326, 6.1379275, -34.4493408, 34.4826126
26: -43.2353592, 7.8830490, -42.9796143, 7.8628988, -44.0469666, 43.7905350
27: -26.7182198, 11.4920158, -26.6781712, 11.5457802, -37.9179382, 37.8203201
28: -29.6671715, 7.1944838, -29.6184731, 7.2091866, -36.7148972, 36.7009583
29: -32.6818771, 9.0342979, -32.6326218, 9.0426865, -41.7245636, 41.6669197
30: -37.7091942, 7.0080357, -37.6117973, 7.0081959, -44.7173920, 44.6198349
31: -31.5457935, 7.3024583, -31.4509544, 7.3115187, -37.7090530, 37.7725906
32: -33.7941017, 6.7851458, -33.7368927, 6.9060888, -40.7001915, 40.5220375
33: -44.0175095, 16.1351242, -44.0203590, 16.0653038, -57.2320023, 57.3883820
34: -50.7626839, -4.1640716, -50.7685928, -4.1585302, -42.3696060, 42.4726257
35: -41.0174217, 7.1797404, -41.0253143, 7.1247697, -43.9353027, 44.0913239
36: -44.4990768, 5.4720869, -44.5048027, 5.5611634, -45.6752930, 45.6371155
37: -59.5346527, 2.4083223, -59.4833832, 2.4575205, -55.3190002, 55.2897263
38: -50.9748726, 8.6620998, -50.9640427, 8.7318716, -59.7067451, 59.6261444
39: -52.2277946, 14.9557734, -52.2086868, 14.9588518, -67.1866455, 67.1644592
40: -47.9360619, 8.3640184, -47.9158096, 8.4410067, -53.3581696, 53.2451859
41: -31.9492569, 15.2890139, -31.9252357, 15.3990498, -45.7421112, 45.5918350
42: -27.2789307, 10.2148867, -27.2171764, 10.2947254, -37.0825195, 36.8849602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1627

## Relational analysis of IS_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109638, upper bound: 14.7966450
time: 40.16 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109638, upper bound: 14.8109642
time: 36.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 78.89 seconds
IS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6429702, upper bound: 14.8109642
IS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6573261, upper bound: 14.8109642
IS_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6429702, upper bound: 14.8109642
IS_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.7021990, upper bound: 14.8109642
IS_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6878400, upper bound: 14.8109641
IS_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.7312643, upper bound: 14.8109641
IS_A1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6720689, upper bound: 14.8109641
IS_A1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.7242453, upper bound: 14.8109641
IS_A1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6720689, upper bound: 14.8109641
IS_A1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.7690824, upper bound: 14.8109641
IS_A2_B2_A1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6640504, upper bound: 14.7966451
IS_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6640504, upper bound: 14.8109643
IS_A2_B2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6640504, upper bound: 14.7966451
IS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.6640504, upper bound: 14.8109643
IS_A2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.7198602, upper bound: 14.8109643
IS_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.7632553, upper bound: 14.8109643
IS_A2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.7819378, upper bound: 14.7966450
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.7819378, upper bound: 14.8109642
IS_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.8109638, upper bound: 14.7518362
IS_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.8109638, upper bound: 14.7661503
IS_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.8109638, upper bound: 14.7966450
IS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 78.89
Output dim: 2, lower bound: -14.8109638, upper bound: 14.8109642

## BFS IS instance: IS_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -23.4104710, 32.6560745, -23.7462978, 32.9596024, -53.9461975, 53.9641876
1: -7.3944626, 32.2202492, -7.5787592, 32.3403625, -36.2353210, 36.3082619
2: -4.4056139, 31.6051311, -4.6311960, 31.8585205, -32.9043579, 32.8595924
3: -8.6635389, 28.7336769, -8.8495464, 29.0234680, -32.3165741, 32.2079659
4: -9.6915483, 34.8724670, -9.9562721, 35.1148529, -43.0475082, 43.0734482
5: -10.8165607, 29.6972504, -11.0367107, 30.0473251, -37.9779510, 37.8361435
6: -38.5182648, 7.1750879, -38.6429291, 7.3635521, -44.0548630, 44.1066437
7: -14.9182558, 30.4940453, -15.1798153, 30.6719971, -41.4149323, 41.4725723
8: -15.0035896, 34.3101158, -15.2779064, 34.6078720, -46.7092438, 46.6715240
9: -10.0267868, 26.9731407, -10.2849178, 27.1410275, -35.4195862, 35.5072975
10: -28.0259171, 23.5788784, -28.5271282, 23.8139496, -49.9839859, 50.2524033
11: -35.4257088, 13.9567366, -35.9465752, 14.1499777, -48.8999710, 49.2368164
12: -49.0804863, 1.4322810, -49.6124153, 1.7940469, -43.5259933, 43.7177353
13: -28.6487083, 21.1118050, -28.7420883, 21.3312302, -49.3642731, 49.2318573
14: -70.5396576, -6.6989021, -71.0436325, -6.5287704, -64.0108871, 64.3447266
15: -17.1487427, 24.4898605, -17.3204880, 24.7159386, -41.8646812, 41.8103485
16: -27.1037159, 23.4397469, -27.4828377, 23.5725937, -47.9928436, 48.1846848
17: -70.9834290, -4.1198101, -71.2348175, -4.0307312, -66.9526978, 67.1150055
18: -34.5972633, 11.5640488, -34.8626747, 11.6177387, -40.3711395, 40.5593872
19: -25.5271034, 5.1557732, -25.7506905, 5.2122064, -29.7006836, 29.8355560
20: -26.2918720, 4.1972742, -26.5206966, 4.2998056, -28.8691254, 29.0044785
21: -31.0961266, 9.9223061, -31.4729767, 10.0473843, -40.0075302, 40.2939148
22: -33.4441643, 6.7549777, -33.6419563, 6.8960242, -38.2107773, 38.2878799
23: -26.7323132, 8.7322140, -26.9745293, 8.8297329, -35.0453873, 35.1363602
24: -23.0859566, 9.7731056, -23.2453136, 9.8152866, -32.5883293, 32.6723175
25: -29.0513248, 5.8791671, -29.2292023, 6.0056186, -34.1484375, 34.1508751
26: -42.7069054, 7.5316238, -43.1710701, 7.7310481, -43.3861465, 43.6706848
27: -26.4560490, 11.3499699, -26.6218014, 11.4100895, -37.4826736, 37.5874443
28: -29.4610310, 6.9782782, -29.6140137, 7.0916324, -36.4103622, 36.4075394
29: -32.4536209, 8.7624187, -32.6359062, 8.8933296, -41.3469505, 41.3983231
30: -37.4173584, 6.7216358, -37.6649399, 6.8795805, -44.2969398, 44.3865738
31: -31.1894894, 7.1833649, -31.4679279, 7.2282791, -37.3834114, 37.5090866
32: -33.4645042, 6.4713449, -33.6413689, 6.6464977, -40.1110001, 40.1127129
33: -43.6441078, 15.7704792, -43.8291321, 16.0486259, -56.9057083, 56.7945480
34: -50.5054169, -4.5161438, -50.6052780, -4.2655191, -42.0851212, 41.9472198
35: -40.7312050, 6.8102393, -40.8856659, 7.0894051, -43.6819229, 43.5690308
36: -44.2887192, 5.1721392, -44.3657913, 5.3552313, -45.2837830, 45.1841507
37: -59.1971931, 2.2204919, -59.3755035, 2.3338780, -54.8854980, 54.9454041
38: -50.6710739, 8.3922634, -50.7970428, 8.5597305, -59.2308044, 59.1893082
39: -51.8982239, 14.7817068, -52.0399055, 14.9012299, -66.7994537, 66.8216095
40: -47.6433029, 8.2172489, -47.7615395, 8.3242054, -52.9342346, 52.9386673
41: -31.6380138, 15.0418367, -31.7884369, 15.1791935, -45.1896057, 45.2347794
42: -26.9621754, 9.8608065, -27.1620445, 10.0622511, -36.3972626, 36.5132751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6429702, upper bound: 14.7877928
time: 40.29 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6429702, upper bound: 14.8109642
time: 42.76 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.4121609, 32.6637001, -23.7994862, 32.9885674, -53.9775391, 54.0206528
1: -7.3945456, 32.2257385, -7.6171732, 32.3634262, -36.2580872, 36.3533783
2: -4.4062243, 31.6108341, -4.6751623, 31.8826523, -32.9260406, 32.9100342
3: -8.6647243, 28.7474632, -8.9188690, 29.0646706, -32.3467865, 32.2854538
4: -9.6927500, 34.8769150, -9.9921217, 35.1373215, -43.0711670, 43.1158142
5: -10.8168678, 29.7085915, -11.0970545, 30.0819740, -38.0069427, 37.9073792
6: -38.5271683, 7.1762075, -38.6818619, 7.4076424, -44.1100464, 44.1457596
7: -14.9188128, 30.5007744, -15.2262030, 30.6967163, -41.4371033, 41.5328293
8: -15.0047445, 34.3163414, -15.3435936, 34.6421890, -46.7433472, 46.7454224
9: -10.0262270, 26.9762707, -10.2948322, 27.1680584, -35.4367828, 35.5450096
10: -28.0428200, 23.5843010, -28.5799274, 23.9304962, -50.0732651, 50.2950287
11: -35.4267311, 13.9569426, -35.9667778, 14.1580820, -48.9123383, 49.2628632
12: -49.0925674, 1.4361639, -49.6592941, 1.8986597, -43.6477203, 43.7613678
13: -28.6528473, 21.1181984, -28.7696400, 21.3916512, -49.4346085, 49.2685242
14: -70.5413208, -6.6910591, -71.0846558, -6.4961357, -64.0451813, 64.3936005
15: -17.1496162, 24.4952011, -17.3635330, 24.7387676, -41.8883820, 41.8587341
16: -27.1055069, 23.4427795, -27.5208797, 23.5993633, -48.0143356, 48.2864075
17: -70.9849701, -4.0978279, -71.3128052, -3.9575386, -67.0274353, 67.2149811
18: -34.6006241, 11.5674820, -34.9153214, 11.6359787, -40.3877640, 40.6138573
19: -25.5307045, 5.1557693, -25.7755394, 5.2271547, -29.7164917, 29.8712082
20: -26.2990017, 4.1983347, -26.5495396, 4.3345189, -28.9019775, 29.0317688
21: -31.0989647, 9.9225340, -31.4977493, 10.0538330, -40.0211716, 40.3249512
22: -33.4464874, 6.7551155, -33.6688080, 6.9079151, -38.2252121, 38.3341904
23: -26.7352200, 8.7324343, -26.9963493, 8.8407650, -35.0606232, 35.1622849
24: -23.0903053, 9.7743063, -23.2742310, 9.8244677, -32.6022797, 32.7022972
25: -29.0533333, 5.8809900, -29.2503471, 6.0313964, -34.1718597, 34.1783981
26: -42.7111816, 7.5328341, -43.1961594, 7.7474008, -43.4075394, 43.7026520
27: -26.4619904, 11.3592949, -26.6937447, 11.4416647, -37.5186462, 37.6625900
28: -29.4647865, 6.9813495, -29.6523132, 7.1180615, -36.4426193, 36.4690018
29: -32.4548836, 8.7680149, -32.6808701, 8.9191685, -41.3740540, 41.4488831
30: -37.4187698, 6.7217360, -37.6801338, 6.8954048, -44.3141747, 44.4018707
31: -31.1956978, 7.1832442, -31.5027103, 7.2413764, -37.3997650, 37.5471725
32: -33.4806900, 6.4728994, -33.6936646, 6.7337513, -40.2144394, 40.1665649
33: -43.6635017, 15.7718525, -43.8950424, 16.1394978, -57.0184479, 56.8600769
34: -50.5214996, -4.5161400, -50.6567001, -4.2198591, -42.1383286, 41.9872437
35: -40.7372360, 6.8095975, -40.9136734, 7.0942755, -43.7034149, 43.5953445
36: -44.2954750, 5.1718030, -44.3980980, 5.3747396, -45.3133392, 45.2167435
37: -59.2155609, 2.2212520, -59.4402428, 2.4173932, -54.9920349, 55.0084305
38: -50.6909103, 8.3926296, -50.8707008, 8.6400414, -59.3309517, 59.2633286
39: -51.9267273, 14.7829056, -52.1320343, 15.0240536, -66.9507828, 66.9149399
40: -47.6574821, 8.2173634, -47.8150063, 8.3986473, -53.0203705, 52.9890518
41: -31.6508675, 15.0426025, -31.8352108, 15.2367773, -45.2571106, 45.2773361
42: -26.9725418, 9.8626795, -27.2032356, 10.1317158, -36.4796371, 36.5511093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=225, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6429702, upper bound: 14.7877928
time: 34.23 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6429702, upper bound: 14.8109642
time: 44.43 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -23.6243153, 32.7147217, -23.8393936, 32.9628830, -54.1585312, 54.1184769
1: -7.5338936, 32.2624054, -7.6398821, 32.3428726, -36.3675232, 36.4164734
2: -4.5384855, 31.6427689, -4.6897078, 31.8617802, -33.0268097, 32.9607315
3: -8.7635918, 28.7676697, -8.8929996, 29.0278664, -32.3984833, 32.2934952
4: -9.8695002, 34.9211693, -10.0336008, 35.1182251, -43.2157593, 43.2013397
5: -10.9317951, 29.7428989, -11.0869884, 30.0529900, -38.0814056, 37.9403915
6: -38.5621529, 7.2755241, -38.6495743, 7.4065351, -44.1488495, 44.2032318
7: -15.1048565, 30.5423222, -15.2614574, 30.6743469, -41.5708160, 41.6147308
8: -15.1824646, 34.3655777, -15.3552694, 34.6134377, -46.8824158, 46.8120499
9: -10.1667051, 27.0180206, -10.3458147, 27.1450729, -35.5431900, 35.6175461
10: -28.1311970, 23.6355019, -28.5725727, 23.8232975, -50.0900574, 50.3552933
11: -35.4599991, 13.9970322, -35.9603500, 14.1606274, -48.9450378, 49.2959824
12: -49.1328201, 1.6062469, -49.6173325, 1.8678408, -43.6676331, 43.8430405
13: -28.6866283, 21.1447010, -28.7577572, 21.3434830, -49.4149170, 49.3034592
14: -70.6793060, -6.6444397, -71.1053925, -6.5207748, -64.1585312, 64.4609528
15: -17.2233391, 24.5219002, -17.3511925, 24.7210255, -41.9443665, 41.8730927
16: -27.2613487, 23.4906654, -27.5514183, 23.5779343, -48.1379318, 48.3079071
17: -71.0329666, -4.0857258, -71.2555237, -4.0241737, -67.0087891, 67.1697998
18: -34.6222649, 11.5945206, -34.8705025, 11.6298943, -40.3981323, 40.6108055
19: -25.5612335, 5.1697178, -25.7610893, 5.2180214, -29.7508659, 29.8606415
20: -26.3174515, 4.2136374, -26.5275154, 4.3075328, -28.9346657, 29.0213852
21: -31.1405296, 9.9409704, -31.4879532, 10.0548429, -40.0714951, 40.3206711
22: -33.4929428, 6.8147583, -33.6536942, 6.9218464, -38.2874680, 38.3477402
23: -26.7615795, 8.7686319, -26.9822998, 8.8447933, -35.1016312, 35.1833267
24: -23.1111202, 9.7910042, -23.2544022, 9.8244171, -32.6265678, 32.6999283
25: -29.0934830, 5.9568634, -29.2382374, 6.0396948, -34.2267990, 34.2348785
26: -42.7517548, 7.5793486, -43.1804810, 7.7517319, -43.4526520, 43.7137222
27: -26.4872208, 11.3867636, -26.6323757, 11.4211311, -37.5294571, 37.6399307
28: -29.4982853, 7.0469670, -29.6215782, 7.1215243, -36.4794235, 36.4843597
29: -32.4943886, 8.8224678, -32.6464844, 8.9194584, -41.4138489, 41.4689522
30: -37.4401398, 6.7582407, -37.6736374, 6.8933105, -44.3334503, 44.4318771
31: -31.2350082, 7.2139392, -31.4794636, 7.2407360, -37.4715500, 37.5515823
32: -33.5073662, 6.5621166, -33.6486359, 6.6867332, -40.1940994, 40.2107544
33: -43.7210617, 15.8607168, -43.8467941, 16.0884724, -57.0239563, 56.8869705
34: -50.5563164, -4.4043708, -50.6120033, -4.2166076, -42.1989441, 42.0264969
35: -40.7904320, 6.9209661, -40.8951836, 7.1381245, -43.8026428, 43.6488953
36: -44.3443680, 5.2979603, -44.3730392, 5.4106436, -45.4034882, 45.2909851
37: -59.2653885, 2.2872562, -59.3909874, 2.3628263, -54.9933014, 54.9757004
38: -50.7458191, 8.4973965, -50.8081627, 8.6050005, -59.3508186, 59.3055573
39: -51.9762726, 14.8055038, -52.0607376, 14.9109831, -66.8872528, 66.8662415
40: -47.7028542, 8.2504044, -47.7755547, 8.3379097, -53.0055695, 52.9809265
41: -31.6869621, 15.1167116, -31.7989750, 15.2112141, -45.2702866, 45.3122787
42: -26.9985046, 9.9494152, -27.1678715, 10.1001053, -36.4900284, 36.5976791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6878397, upper bound: 14.7661502
time: 55.27 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6429702, upper bound: 14.7662030
time: 38.63 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.6260147, 32.7223320, -23.8925266, 32.9919052, -54.1898117, 54.1749496
1: -7.5339699, 32.2679100, -7.6782856, 32.3659286, -36.3902588, 36.4615555
2: -4.5391092, 31.6484699, -4.7336903, 31.8859138, -33.0484467, 33.0111694
3: -8.7647915, 28.7814255, -8.9622936, 29.0690670, -32.4287033, 32.3710136
4: -9.8706951, 34.9256210, -10.0694790, 35.1406670, -43.2393951, 43.2437363
5: -10.9320755, 29.7542591, -11.1472988, 30.0876465, -38.1103287, 38.0116882
6: -38.5710869, 7.2766352, -38.6885147, 7.4506168, -44.2039948, 44.2423019
7: -15.1053820, 30.5490799, -15.3078833, 30.6990280, -41.5930099, 41.6749725
8: -15.1835613, 34.3718491, -15.4209728, 34.6477890, -46.9164886, 46.8859940
9: -10.1661425, 27.0211449, -10.3557529, 27.1721153, -35.5603790, 35.6552582
10: -28.1480713, 23.6409492, -28.6253643, 23.9398956, -50.1793671, 50.3980103
11: -35.4610367, 13.9971790, -35.9805832, 14.1687803, -48.9573669, 49.3220215
12: -49.1449203, 1.6100812, -49.6641541, 1.9724021, -43.7893372, 43.8865967
13: -28.6907272, 21.1511192, -28.7853470, 21.4039383, -49.4852753, 49.3401566
14: -70.6809845, -6.6365795, -71.1464081, -6.4881248, -64.1928558, 64.5098267
15: -17.2241974, 24.5272865, -17.3942108, 24.7438526, -41.9680481, 41.9214973
16: -27.2630959, 23.4936733, -27.5894337, 23.6047211, -48.1594391, 48.4095383
17: -71.0345612, -4.0637360, -71.3334808, -3.9509926, -67.0835724, 67.2697449
18: -34.6255989, 11.5979748, -34.9232063, 11.6481552, -40.4147873, 40.6652679
19: -25.5648499, 5.1697454, -25.7859306, 5.2329526, -29.7666588, 29.8963051
20: -26.3245926, 4.2147260, -26.5563202, 4.3422399, -28.9674835, 29.0486679
21: -31.1434097, 9.9411850, -31.5127010, 10.0612221, -40.0851288, 40.3516922
22: -33.4952927, 6.8149223, -33.6805420, 6.9337354, -38.3019257, 38.3940659
23: -26.7644691, 8.7688799, -27.0041103, 8.8558578, -35.1168823, 35.2093163
24: -23.1154671, 9.7921677, -23.2833023, 9.8336086, -32.6405487, 32.7298775
25: -29.0954952, 5.9586978, -29.2593765, 6.0654402, -34.2502441, 34.2624512
26: -42.7560425, 7.5805364, -43.2055588, 7.7681417, -43.4740829, 43.7457504
27: -26.4931602, 11.3961143, -26.7043343, 11.4526787, -37.5654678, 37.7150841
28: -29.5020561, 7.0500011, -29.6598892, 7.1479635, -36.5116806, 36.5459213
29: -32.4956589, 8.8280430, -32.6914444, 8.9452715, -41.4409294, 41.5194855
30: -37.4415817, 6.7583504, -37.6888847, 6.9091187, -44.3507004, 44.4472351
31: -31.2412071, 7.2137952, -31.5142365, 7.2537980, -37.4879494, 37.5896645
32: -33.5235443, 6.5636511, -33.7009239, 6.7739811, -40.2975235, 40.2645760
33: -43.7403946, 15.8620567, -43.9127197, 16.1792984, -57.1367035, 56.9524689
34: -50.5724030, -4.4043803, -50.6633568, -4.1709471, -42.2521973, 42.0665359
35: -40.7964134, 6.9203019, -40.9232140, 7.1429510, -43.8241196, 43.6752090
36: -44.3511047, 5.2976265, -44.4053879, 5.4300961, -45.4330750, 45.3235245
37: -59.2836456, 2.2880096, -59.4557190, 2.4462895, -55.0998993, 55.0386810
38: -50.7657166, 8.4977312, -50.8818016, 8.6852732, -59.4509888, 59.3795319
39: -52.0047150, 14.8066406, -52.1528778, 15.0337601, -67.0384750, 66.9595184
40: -47.7170334, 8.2505236, -47.8290520, 8.4123478, -53.0916748, 53.0313110
41: -31.6998482, 15.1174307, -31.8457966, 15.2688017, -45.3377991, 45.3548965
42: -27.0089092, 9.9513073, -27.2090797, 10.1695404, -36.5723343, 36.6354980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=225, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7021987, upper bound: 14.7661502
time: 34.25 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6573261, upper bound: 14.7662029
time: 24.67 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -23.6305103, 32.7223282, -23.9090595, 32.9926071, -54.1993866, 54.1916885
1: -7.5372610, 32.2707253, -7.6886635, 32.3702240, -36.4048233, 36.4649391
2: -4.5426941, 31.6589546, -4.7800274, 31.9102325, -33.0910263, 33.0530052
3: -8.7672138, 28.7705612, -8.9393148, 29.0436974, -32.4358521, 32.3253784
4: -9.8741550, 34.9369278, -10.1226387, 35.1677017, -43.2823181, 43.2915726
5: -10.9356794, 29.7475567, -11.1310978, 30.0749893, -38.1232529, 37.9802628
6: -38.5902748, 7.2786255, -38.7369003, 7.5388861, -44.3050842, 44.2923508
7: -15.1094780, 30.5538845, -15.3187485, 30.7088661, -41.6424255, 41.6504440
8: -15.1874447, 34.3813400, -15.4343176, 34.6628494, -46.9467316, 46.8940048
9: -10.1665134, 27.0202770, -10.3722572, 27.1787605, -35.6049500, 35.6253128
10: -28.1350956, 23.6473942, -28.6499481, 23.8746719, -50.1467590, 50.4836502
11: -35.4624634, 14.0070591, -36.0068130, 14.1999750, -48.9902344, 49.3489990
12: -49.1431122, 1.6109138, -49.6553116, 1.9089494, -43.6737289, 43.9415970
13: -28.6903687, 21.1458035, -28.7984848, 21.3702850, -49.4545746, 49.3431625
14: -70.6836624, -6.6187592, -71.2443542, -6.4457207, -64.2379456, 64.6255951
15: -17.2261333, 24.5244923, -17.3835258, 24.7372475, -41.9633789, 41.9080200
16: -27.2652130, 23.4940414, -27.6062889, 23.6041489, -48.1875839, 48.3489838
17: -71.0387955, -4.0452118, -71.4455414, -3.9114647, -67.1273346, 67.4003296
18: -34.6242828, 11.5935249, -34.9129868, 11.6531143, -40.4314804, 40.6470451
19: -25.5634289, 5.1733832, -25.7940559, 5.2405891, -29.7669182, 29.8986511
20: -26.3179779, 4.2153411, -26.5375957, 4.3301597, -28.9354897, 29.0575218
21: -31.1421146, 9.9440184, -31.5175362, 10.0778942, -40.0866547, 40.3571777
22: -33.4954758, 6.8185959, -33.6882362, 6.9460044, -38.2984467, 38.4202805
23: -26.7634563, 8.7721348, -27.0081825, 8.8679619, -35.1214294, 35.2179642
24: -23.1131783, 9.7970963, -23.3103390, 9.8602734, -32.6598969, 32.7634201
25: -29.0944576, 5.9630017, -29.2865715, 6.0731535, -34.2595673, 34.2990265
26: -42.7540092, 7.5797644, -43.2056808, 7.7637787, -43.4594727, 43.7588272
27: -26.4966164, 11.3881664, -26.6778374, 11.4846134, -37.6246262, 37.6882095
28: -29.4995308, 7.0505342, -29.6395493, 7.1625142, -36.5260620, 36.5069504
29: -32.4960022, 8.8264523, -32.6769638, 8.9432240, -41.4392242, 41.5034180
30: -37.4406433, 6.7616444, -37.6875687, 6.9212952, -44.3619385, 44.4492111
31: -31.2382431, 7.2246485, -31.5173016, 7.2804127, -37.4906616, 37.6088371
32: -33.5350685, 6.5641813, -33.7384300, 6.8223381, -40.3574066, 40.3026123
33: -43.7244873, 15.8631153, -43.8700562, 16.1311684, -57.0674744, 56.9367752
34: -50.5736923, -4.4017930, -50.6733971, -4.1278296, -42.2711105, 42.1331177
35: -40.7997513, 6.9235663, -40.9319267, 7.1839209, -43.8240509, 43.7270660
36: -44.3725891, 5.3008595, -44.4603386, 5.5314703, -45.5242844, 45.4087448
37: -59.2773094, 2.2900758, -59.4370804, 2.4447336, -55.0655975, 55.0577393
38: -50.7673149, 8.4997530, -50.8886909, 8.7088680, -59.4761810, 59.3884430
39: -51.9851227, 14.8071995, -52.1039124, 14.9584332, -66.9435577, 66.9111099
40: -47.7255096, 8.2522936, -47.8471069, 8.4557314, -53.1425476, 53.0560455
41: -31.7128010, 15.1196136, -31.8844337, 15.3426437, -45.4236450, 45.4036331
42: -27.0164604, 9.9519501, -27.2300911, 10.1992664, -36.5951691, 36.6710243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=176, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6720688, upper bound: 14.8109638
time: 70.67 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6720688, upper bound: 14.7662029
time: 15.94 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -23.6321793, 32.7299271, -23.9621639, 33.0215836, -54.2306976, 54.2481689
1: -7.5373793, 32.2762108, -7.7270727, 32.3932953, -36.4275818, 36.5100975
2: -4.5433216, 31.6646652, -4.8239651, 31.9344692, -33.1127396, 33.1034851
3: -8.7684002, 28.7843552, -9.0086250, 29.0848866, -32.4660721, 32.4029007
4: -9.8753443, 34.9414024, -10.1585112, 35.1902695, -43.3059616, 43.3339386
5: -10.9359674, 29.7588882, -11.1914339, 30.1096191, -38.1522369, 38.0515900
6: -38.5991898, 7.2797480, -38.7758942, 7.5829506, -44.3601913, 44.3314896
7: -15.1100168, 30.5605946, -15.3651295, 30.7335796, -41.6646652, 41.7107010
8: -15.1885662, 34.3876190, -15.4999676, 34.6971703, -46.9808350, 46.9679871
9: -10.1659670, 27.0234013, -10.3821440, 27.2058010, -35.6221161, 35.6629944
10: -28.1520061, 23.6528244, -28.7027683, 23.9912930, -50.2360764, 50.5263443
11: -35.4635010, 14.0072460, -36.0270462, 14.2080793, -49.0025864, 49.3750687
12: -49.1552811, 1.6147795, -49.7021904, 2.0135217, -43.7954865, 43.9851379
13: -28.6944637, 21.1522026, -28.8260994, 21.4306927, -49.5249023, 49.3798447
14: -70.6853790, -6.6108990, -71.2853546, -6.4130516, -64.2723236, 64.6744537
15: -17.2270107, 24.5298576, -17.4265442, 24.7600822, -41.9870911, 41.9564018
16: -27.2670059, 23.4970531, -27.6442928, 23.6308784, -48.2090454, 48.4506226
17: -71.0403366, -4.0232277, -71.5234833, -3.8383503, -67.2019882, 67.5002594
18: -34.6276054, 11.5969458, -34.9657135, 11.6713333, -40.4481506, 40.7014923
19: -25.5670414, 5.1734204, -25.8189640, 5.2555132, -29.7827148, 29.9342766
20: -26.3250809, 4.2164321, -26.5664368, 4.3648648, -28.9682999, 29.0848160
21: -31.1449833, 9.9442577, -31.5423336, 10.0843124, -40.1002960, 40.3882294
22: -33.4977913, 6.8187451, -33.7151222, 6.9578738, -38.3128738, 38.4666748
23: -26.7663803, 8.7723494, -27.0300236, 8.8790121, -35.1367188, 35.2439270
24: -23.1175385, 9.7982969, -23.3392811, 9.8694363, -32.6739082, 32.7933540
25: -29.0964737, 5.9648490, -29.3077354, 6.0988827, -34.2829895, 34.3265762
26: -42.7582626, 7.5809746, -43.2307892, 7.7801561, -43.4808807, 43.7908554
27: -26.5024796, 11.3974962, -26.7497902, 11.5161448, -37.6606522, 37.7633057
28: -29.5032692, 7.0535979, -29.6778908, 7.1889696, -36.5583038, 36.5684433
29: -32.4972725, 8.8320084, -32.7219734, 8.9690428, -41.4663162, 41.5539818
30: -37.4420662, 6.7617359, -37.7027817, 6.9371004, -44.3791656, 44.4645157
31: -31.2444134, 7.2244844, -31.5520935, 7.2934976, -37.5070343, 37.6469498
32: -33.5512695, 6.5657187, -33.7907982, 6.9095716, -40.4608421, 40.3565178
33: -43.7438774, 15.8644428, -43.9360008, 16.2219887, -57.1802139, 57.0023117
34: -50.5898132, -4.4017529, -50.7248688, -4.0821333, -42.3243561, 42.1732635
35: -40.8057098, 6.9228687, -40.9599266, 7.1887584, -43.8455734, 43.7534409
36: -44.3793640, 5.3005414, -44.4928207, 5.5509281, -45.5539246, 45.4414520
37: -59.2956123, 2.2907844, -59.5018082, 2.5281906, -55.1720886, 55.1207504
38: -50.7872276, 8.5001059, -50.9623032, 8.7892437, -59.5764694, 59.4624100
39: -52.0135574, 14.8083553, -52.1960754, 15.0811930, -67.0947495, 67.0044327
40: -47.7396660, 8.2524452, -47.9006042, 8.5301790, -53.2286682, 53.1064301
41: -31.7256260, 15.1203480, -31.9313793, 15.4002037, -45.4911652, 45.4462891
42: -27.0268421, 9.9538746, -27.2712898, 10.2687092, -36.6774521, 36.7088776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=176, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=225, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6864035, upper bound: 14.8109638
time: 45.12 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6864035, upper bound: 14.7662029
time: 25.66 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -23.6449528, 32.7345695, -23.8389473, 32.9901009, -54.1847076, 54.1343689
1: -7.5273533, 32.2684517, -7.6246777, 32.3709412, -36.3771973, 36.3984413
2: -4.6085215, 31.6783180, -4.6944942, 31.9046440, -33.1344223, 32.9936905
3: -8.7862511, 28.7926922, -8.8939304, 29.0401249, -32.4438705, 32.3081779
4: -9.9059010, 34.9570122, -10.0229759, 35.1616554, -43.3013306, 43.2249527
5: -10.9418430, 29.7702484, -11.0830650, 30.0723171, -38.1248550, 37.9524384
6: -38.6129494, 7.4537659, -38.7207031, 7.4415474, -44.2281952, 44.4566116
7: -15.0744381, 30.5552368, -15.2341614, 30.7056084, -41.5795288, 41.5875092
8: -15.2346802, 34.4147415, -15.3583860, 34.6566658, -46.9674225, 46.8542938
9: -10.1324816, 27.0399399, -10.3303566, 27.1530228, -35.5545502, 35.6431694
10: -28.1746769, 23.6498566, -28.5697403, 23.8653393, -50.2419510, 50.4169006
11: -35.5277786, 14.0716848, -35.9653320, 14.2205181, -49.0990372, 49.3794250
12: -49.1559639, 1.5885587, -49.6467133, 1.8558831, -43.6689758, 43.8909378
13: -28.7659111, 21.1883183, -28.7894821, 21.3522644, -49.5204315, 49.3816147
14: -70.8363113, -6.6040668, -71.1311798, -6.4496231, -64.3866882, 64.5271149
15: -17.2588806, 24.5346527, -17.3630562, 24.7343674, -41.9932480, 41.8977089
16: -27.2388973, 23.4885330, -27.5267639, 23.5909786, -48.1510315, 48.2883835
17: -71.2131958, -4.0409012, -71.2707901, -3.9158344, -67.2973633, 67.2298889
18: -34.6744690, 11.6359711, -34.8809700, 11.6534882, -40.4905968, 40.6611328
19: -25.6051140, 5.2192192, -25.7702179, 5.2486076, -29.8164597, 29.9221802
20: -26.3388672, 4.2848482, -26.5325394, 4.3378563, -28.9569855, 29.0910301
21: -31.1833267, 10.0284739, -31.4931355, 10.1048431, -40.1578140, 40.4210434
22: -33.5376549, 6.8709345, -33.6608810, 6.9549770, -38.3874664, 38.3919907
23: -26.8043175, 8.8084555, -26.9917030, 8.8698750, -35.1653442, 35.2351913
24: -23.1889896, 9.8428612, -23.2639275, 9.8591461, -32.7417145, 32.7583923
25: -29.1495762, 5.9801555, -29.2441082, 6.0642333, -34.3122787, 34.2667465
26: -42.7968330, 7.6389484, -43.1904144, 7.7821660, -43.5386658, 43.7793427
27: -26.5383415, 11.4816628, -26.6622086, 11.4479637, -37.6204605, 37.7811050
28: -29.5301056, 7.0970278, -29.6331882, 7.1387229, -36.5268250, 36.5514145
29: -32.5367203, 8.8921385, -32.6541519, 8.9560652, -41.4927864, 41.5462914
30: -37.4914169, 6.8611526, -37.6766434, 6.9490347, -44.4404526, 44.5377960
31: -31.2702713, 7.2386956, -31.4901466, 7.2722292, -37.5175323, 37.5876579
32: -33.5565872, 6.6929703, -33.7259064, 6.6927977, -40.2493858, 40.4188766
33: -43.7369232, 15.8584213, -43.8658257, 16.0756950, -57.0416412, 56.9431458
34: -50.5922394, -4.3456507, -50.6684418, -4.2205038, -42.2247162, 42.1723328
35: -40.8114090, 6.9131427, -40.9265060, 7.1198578, -43.7967377, 43.7197266
36: -44.3817673, 5.3790426, -44.4489250, 5.4026937, -45.4262848, 45.4580078
37: -59.2976532, 2.3402014, -59.4323044, 2.3606987, -55.0151672, 55.1260757
38: -50.7654343, 8.5459175, -50.8690681, 8.5894794, -59.3549118, 59.4149857
39: -52.0039406, 14.8424034, -52.0932312, 14.9136686, -66.9176102, 66.9356384
40: -47.7251358, 8.3666496, -47.8392258, 8.3412352, -53.0337524, 53.1926498
41: -31.7381153, 15.2658920, -31.8697701, 15.2305784, -45.3429642, 45.5374374
42: -27.0352936, 10.0458527, -27.2192688, 10.1100483, -36.5178680, 36.7534714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=176, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6720689, upper bound: 14.7877927
time: 38.92 seconds

## Relational analysis of IS_A1_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6720689, upper bound: 14.8109641
time: 54.29 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -23.6466579, 32.7422485, -23.8921356, 33.0191345, -54.2159882, 54.1909027
1: -7.5274258, 32.2739487, -7.6630831, 32.3940201, -36.3999557, 36.4435730
2: -4.6091366, 31.6840248, -4.7384624, 31.9288177, -33.1560669, 33.0441780
3: -8.7874537, 28.8064804, -8.9632215, 29.0813255, -32.4740982, 32.3856506
4: -9.9071198, 34.9614792, -10.0588312, 35.1841469, -43.3249817, 43.2673264
5: -10.9421539, 29.7816010, -11.1433811, 30.1069527, -38.1538315, 38.0237122
6: -38.6218605, 7.4549141, -38.7596436, 7.4856262, -44.2833252, 44.4957047
7: -15.0749731, 30.5619869, -15.2805643, 30.7302475, -41.6017303, 41.6477661
8: -15.2357960, 34.4210281, -15.4240894, 34.6909828, -47.0015411, 46.9282684
9: -10.1319056, 27.0430889, -10.3402586, 27.1800785, -35.5717163, 35.6808548
10: -28.1915588, 23.6552696, -28.6225777, 23.9819031, -50.3312531, 50.4596176
11: -35.5288162, 14.0718985, -35.9855957, 14.2287045, -49.1113892, 49.4054871
12: -49.1680679, 1.5924234, -49.6935501, 1.9604526, -43.7907104, 43.9345169
13: -28.7699852, 21.1947136, -28.8170643, 21.4126282, -49.5907898, 49.4182816
14: -70.8379517, -6.5961876, -71.1722488, -6.4169769, -64.4209747, 64.5760651
15: -17.2597656, 24.5400162, -17.4060383, 24.7571754, -42.0169411, 41.9460526
16: -27.2406712, 23.4915810, -27.5647926, 23.6177368, -48.1725540, 48.3900299
17: -71.2147827, -4.0189419, -71.3486938, -3.8426208, -67.3721619, 67.3297501
18: -34.6777992, 11.6394138, -34.9336128, 11.6717138, -40.5071945, 40.7155380
19: -25.6087093, 5.2192316, -25.7950764, 5.2635341, -29.8322525, 29.9578133
20: -26.3459930, 4.2859235, -26.5613766, 4.3725858, -28.9898338, 29.1183281
21: -31.1861649, 10.0287113, -31.5179043, 10.1112356, -40.1713943, 40.4520798
22: -33.5399857, 6.8710856, -33.6877403, 6.9668298, -38.4018555, 38.4382935
23: -26.8071938, 8.8086777, -27.0135365, 8.8809328, -35.1805649, 35.2611618
24: -23.1933479, 9.8440189, -23.2928905, 9.8683186, -32.7556686, 32.7883453
25: -29.1515846, 5.9819679, -29.2652588, 6.0899715, -34.3357162, 34.2943497
26: -42.8010941, 7.6401501, -43.2154846, 7.7985678, -43.5600433, 43.8113937
27: -26.5442448, 11.4909992, -26.7341785, 11.4795151, -37.6564865, 37.8562469
28: -29.5338573, 7.1000414, -29.6715260, 7.1651974, -36.5590820, 36.6128998
29: -32.5380211, 8.8976536, -32.6991425, 8.9818974, -41.5199203, 41.5967941
30: -37.4927979, 6.8612700, -37.6918564, 6.9648676, -44.4576645, 44.5531273
31: -31.2764740, 7.2385507, -31.5248966, 7.2852840, -37.5338860, 37.6257706
32: -33.5727806, 6.6945496, -33.7782173, 6.7800121, -40.3527908, 40.4727669
33: -43.7563400, 15.8597507, -43.9317780, 16.1665916, -57.1544189, 57.0086975
34: -50.6083336, -4.3456397, -50.7198181, -4.1748133, -42.2779999, 42.2124405
35: -40.8174248, 6.9124727, -40.9545135, 7.1246810, -43.8182220, 43.7461472
36: -44.3885345, 5.3786955, -44.4812965, 5.4221830, -45.4559708, 45.4906311
37: -59.3159676, 2.3409104, -59.4970665, 2.4441342, -55.1217194, 55.1890869
38: -50.7853394, 8.5462494, -50.9427376, 8.6697664, -59.4551048, 59.4889870
39: -52.0323334, 14.8435926, -52.1854210, 15.0364628, -67.0687943, 67.0290146
40: -47.7393341, 8.3667870, -47.8927383, 8.4156399, -53.1198807, 53.2430191
41: -31.7509613, 15.2666368, -31.9165955, 15.2881632, -45.4105072, 45.5800095
42: -27.0457058, 10.0477667, -27.2604923, 10.1794863, -36.6002579, 36.7913399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=176, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=225, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7242453, upper bound: 14.7877927
time: 35.55 seconds

## Relational analysis of IS_A1_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7242453, upper bound: 14.8109641
time: 35.48 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -23.8589344, 32.7932549, -23.9320717, 32.9934349, -54.3972168, 54.2886734
1: -7.6667662, 32.3106232, -7.6858025, 32.3734322, -36.5093765, 36.5066566
2: -4.7414355, 31.7159424, -4.7530193, 31.9079285, -33.2568512, 33.0948257
3: -8.8864393, 28.8266754, -8.9373817, 29.0445232, -32.5258942, 32.3937035
4: -10.0838795, 35.0057068, -10.1003513, 35.1650085, -43.4696732, 43.3528900
5: -11.0570831, 29.8158360, -11.1333151, 30.0779266, -38.2282562, 38.0566940
6: -38.6567917, 7.5543947, -38.7273521, 7.4845695, -44.3221512, 44.5532837
7: -15.2609940, 30.6034985, -15.3158350, 30.7078915, -41.7354355, 41.7296448
8: -15.4135628, 34.4703102, -15.4358082, 34.6622887, -47.1405563, 46.9948425
9: -10.2724781, 27.0848522, -10.3912945, 27.1570606, -35.6781998, 35.7534676
10: -28.2800732, 23.7064724, -28.6151867, 23.8746872, -50.3480835, 50.5198288
11: -35.5620422, 14.1119576, -35.9791031, 14.2312346, -49.1441574, 49.4385605
12: -49.2082443, 1.7627096, -49.6515884, 1.9296865, -43.8106613, 44.0164871
13: -28.8039875, 21.2211571, -28.8052311, 21.3645039, -49.5711670, 49.4532089
14: -70.9759903, -6.5496483, -71.1930008, -6.4416790, -64.5343094, 64.6433563
15: -17.3335609, 24.5667114, -17.3937149, 24.7394657, -42.0730286, 41.9604263
16: -27.3965454, 23.5394306, -27.5953369, 23.5962791, -48.2962112, 48.4115372
17: -71.2627411, -4.0067997, -71.2914124, -3.9092903, -67.3534546, 67.2846146
18: -34.6994705, 11.6662979, -34.8888054, 11.6656189, -40.5176392, 40.7125397
19: -25.6393166, 5.2332153, -25.7806282, 5.2544212, -29.8666534, 29.9472656
20: -26.3644314, 4.3012033, -26.5393562, 4.3455520, -29.0225563, 29.1079712
21: -31.2277298, 10.0470486, -31.5081100, 10.1122446, -40.2217636, 40.4477005
22: -33.5865250, 6.9307036, -33.6726189, 6.9808068, -38.4642487, 38.4518661
23: -26.8335686, 8.8449001, -26.9994850, 8.8849602, -35.2216034, 35.2821770
24: -23.2142010, 9.8606949, -23.2730293, 9.8682442, -32.7799835, 32.7859268
25: -29.1917686, 6.0578179, -29.2531605, 6.0982914, -34.3906479, 34.3507729
26: -42.8416405, 7.6867533, -43.1998215, 7.8029222, -43.6052551, 43.8224182
27: -26.5694771, 11.5184546, -26.6727962, 11.4589891, -37.6672134, 37.8335724
28: -29.5673180, 7.1657386, -29.6407261, 7.1685996, -36.5958633, 36.6282883
29: -32.5775452, 8.9522181, -32.6647873, 8.9821873, -41.5597305, 41.6170044
30: -37.5142746, 6.8977184, -37.6853447, 6.9627542, -44.4770279, 44.5830612
31: -31.3158112, 7.2692547, -31.5016575, 7.2846293, -37.6056519, 37.6301804
32: -33.5993652, 6.7839880, -33.7331505, 6.7330046, -40.3323708, 40.5171394
33: -43.8137779, 15.9487629, -43.8834686, 16.1155586, -57.1598358, 57.0357285
34: -50.6430283, -4.2339239, -50.6750755, -4.1715994, -42.3385010, 42.2515945
35: -40.8705864, 7.0239019, -40.9359970, 7.1685648, -43.9174347, 43.7996216
36: -44.4373169, 5.5049057, -44.4561653, 5.4580922, -45.5458832, 45.5648727
37: -59.3657417, 2.4069047, -59.4477501, 2.3895903, -55.1228943, 55.1563339
38: -50.8401413, 8.6510286, -50.8801689, 8.6347485, -59.4748917, 59.5311966
39: -52.0818367, 14.8661871, -52.1140289, 14.9234018, -67.0052414, 66.9802170
40: -47.7846832, 8.3997946, -47.8532715, 8.3549137, -53.1051331, 53.2348175
41: -31.7869949, 15.3408470, -31.8803272, 15.2625599, -45.4235382, 45.6149521
42: -27.0715656, 10.1345158, -27.2250977, 10.1478558, -36.6105499, 36.8378677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=176, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7547536, upper bound: 14.7661501
time: 18.99 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6720689, upper bound: 14.7662028
time: 15.72 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -23.8606262, 32.8008957, -23.9852085, 33.0224495, -54.4284515, 54.3451538
1: -7.6668911, 32.3161125, -7.7241888, 32.3965149, -36.5321503, 36.5517807
2: -4.7420406, 31.7216644, -4.7969780, 31.9321270, -33.2785568, 33.1452866
3: -8.8876362, 28.8404198, -9.0066614, 29.0856895, -32.5560989, 32.4712105
4: -10.0851011, 35.0101700, -10.1361856, 35.1875229, -43.4933167, 43.3952484
5: -11.0573750, 29.8272152, -11.1936369, 30.1125965, -38.2572403, 38.1279678
6: -38.6657143, 7.5554848, -38.7662582, 7.5286293, -44.3772278, 44.5924072
7: -15.2615128, 30.6103058, -15.3622093, 30.7325859, -41.7576828, 41.7899399
8: -15.4146681, 34.4765053, -15.5015030, 34.6965981, -47.1746674, 47.0687256
9: -10.2718925, 27.0879955, -10.4011946, 27.1841164, -35.6953583, 35.7911720
10: -28.2968960, 23.7118759, -28.6680317, 23.9913025, -50.4374542, 50.5625076
11: -35.5631180, 14.1121016, -35.9993172, 14.2393513, -49.1565018, 49.4645920
12: -49.2203674, 1.7665448, -49.6984329, 2.0342393, -43.9323959, 44.0600128
13: -28.8080597, 21.2275696, -28.8328781, 21.4249420, -49.6415024, 49.4899063
14: -70.9776382, -6.5417747, -71.2340240, -6.4090347, -64.5686035, 64.6922455
15: -17.3344460, 24.5720787, -17.4367104, 24.7623119, -42.0967560, 42.0087891
16: -27.3983040, 23.5424461, -27.6333771, 23.6230621, -48.3176651, 48.5132065
17: -71.2643280, -3.9848003, -71.3693466, -3.8361664, -67.4281616, 67.3845444
18: -34.7028198, 11.6697559, -34.9414864, 11.6838627, -40.5342865, 40.7669830
19: -25.6429291, 5.2332220, -25.8054714, 5.2693472, -29.8824463, 29.9828835
20: -26.3715839, 4.3022795, -26.5681648, 4.3802967, -29.0553932, 29.1352921
21: -31.2305927, 10.0472717, -31.5328712, 10.1186752, -40.2353668, 40.4788055
22: -33.5888786, 6.9308405, -33.6994781, 6.9926691, -38.4786453, 38.4982224
23: -26.8364735, 8.8451176, -27.0213432, 8.8960180, -35.2368240, 35.3081741
24: -23.2185364, 9.8618393, -23.3019962, 9.8774300, -32.7939796, 32.8158569
25: -29.1937885, 6.0596685, -29.2743206, 6.1240220, -34.4140854, 34.3783569
26: -42.8459282, 7.6879439, -43.2248688, 7.8193560, -43.6266479, 43.8544617
27: -26.5753899, 11.5277805, -26.7447739, 11.4905586, -37.7032623, 37.9087105
28: -29.5711288, 7.1687570, -29.6790924, 7.1950731, -36.6281052, 36.6898499
29: -32.5788231, 8.9577398, -32.7097397, 9.0080347, -41.5868568, 41.6674805
30: -37.5157089, 6.8978252, -37.7005882, 6.9785986, -44.4943085, 44.5984116
31: -31.3220234, 7.2691269, -31.5364094, 7.2976952, -37.6220474, 37.6682625
32: -33.6155815, 6.7855024, -33.7854996, 6.8202534, -40.4358368, 40.5710030
33: -43.8331833, 15.9501057, -43.9494171, 16.2063770, -57.2725449, 57.1012573
34: -50.6591225, -4.2339177, -50.7265091, -4.1259441, -42.3918304, 42.2917252
35: -40.8765793, 7.0232310, -40.9640427, 7.1734219, -43.9389114, 43.8259888
36: -44.4441071, 5.5045776, -44.4885826, 5.4775577, -45.5755768, 45.5975189
37: -59.3840523, 2.4076681, -59.5125351, 2.4730940, -55.2294769, 55.2193604
38: -50.8600845, 8.6513834, -50.9538307, 8.7150421, -59.5751266, 59.6052132
39: -52.1103363, 14.8673859, -52.2061539, 15.0461159, -67.1564484, 67.0735397
40: -47.7988586, 8.3999300, -47.9067650, 8.4294024, -53.1912155, 53.2852478
41: -31.7998848, 15.3415546, -31.9271736, 15.3201771, -45.4911499, 45.6575775
42: -27.0819817, 10.1363926, -27.2662868, 10.2173176, -36.6928711, 36.8757362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=176, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=225, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7690822, upper bound: 14.7661501
time: 30.70 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7242453, upper bound: 14.7662028
time: 33.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -23.7244053, 32.8197174, -23.8388863, 32.8623924, -54.1175003, 54.1946030
1: -7.5701914, 32.2869492, -7.6305475, 32.3081245, -36.3936386, 36.3816795
2: -4.6478295, 31.7468300, -4.7176838, 31.7528248, -32.9767990, 33.0725212
3: -8.8720570, 28.8790779, -8.9125652, 28.9029140, -32.3392487, 32.3714638
4: -9.9614544, 35.0195541, -10.0620594, 35.0381622, -43.2036896, 43.2973099
5: -11.0562963, 29.8830662, -11.1156578, 29.9148922, -38.0167999, 38.0851059
6: -38.6634369, 7.4068727, -38.6646347, 7.4408531, -44.3716583, 44.3440628
7: -15.1286240, 30.5782089, -15.2266502, 30.6066875, -41.5413208, 41.5781631
8: -15.3282757, 34.5123253, -15.4054852, 34.5319786, -46.8929443, 46.9734802
9: -10.2012691, 27.1588173, -10.2739744, 27.2016850, -35.6980591, 35.6024780
10: -28.3521671, 23.9676628, -28.4057674, 23.9548836, -50.4917068, 50.3169785
11: -35.7626648, 14.2393970, -35.7780991, 14.2970009, -49.3993149, 49.2965698
12: -49.3504524, 1.8679676, -49.3804932, 1.9687948, -43.9438934, 43.8644333
13: -28.7726631, 21.3330574, -28.7920494, 21.3286743, -49.5267715, 49.5269089
14: -70.8867188, -6.4695759, -70.9321136, -6.4107552, -64.4759674, 64.4625397
15: -17.3765068, 24.6511383, -17.4017448, 24.6777534, -42.0542603, 42.0528831
16: -27.3877087, 23.6214333, -27.4542713, 23.6748505, -48.4296417, 48.2668381
17: -71.1734924, -3.9083519, -71.1300964, -3.8934937, -67.2799988, 67.2217407
18: -34.7641602, 11.6281233, -34.7414169, 11.6549911, -40.5788231, 40.5175095
19: -25.6782742, 5.2573175, -25.6945210, 5.2627001, -29.8748779, 29.8980827
20: -26.4460087, 4.3704286, -26.4649124, 4.3688831, -29.0467072, 29.0390244
21: -31.3278599, 10.1125050, -31.3503151, 10.1479568, -40.3657379, 40.2624664
22: -33.5806656, 6.9078698, -33.6106224, 6.9555459, -38.4036407, 38.3920059
23: -26.8908787, 8.8587360, -26.9036350, 8.8779802, -35.1868820, 35.1914749
24: -23.2371025, 9.8405933, -23.2495480, 9.8386278, -32.7607155, 32.8041954
25: -29.1753788, 6.0353918, -29.2043972, 6.0671539, -34.2858734, 34.3092880
26: -42.9130783, 7.7261744, -42.9432068, 7.8168163, -43.6833344, 43.5955505
27: -26.6537590, 11.4372759, -26.6188049, 11.4690466, -37.7463989, 37.6963730
28: -29.5865059, 7.1029983, -29.5903778, 7.1340432, -36.5783386, 36.5577621
29: -32.5986481, 8.9375095, -32.5902481, 8.9877090, -41.5863571, 41.5277557
30: -37.5739555, 6.9347239, -37.5869408, 6.9616432, -44.5355988, 44.5216637
31: -31.3722286, 7.2568078, -31.3966312, 7.2575965, -37.5049782, 37.6400452
32: -33.6458549, 6.7436438, -33.6349106, 6.7297411, -40.3755951, 40.3785553
33: -43.8850403, 16.0139370, -43.9682884, 15.9818249, -57.0148621, 57.1944199
34: -50.6666336, -4.3171887, -50.6944542, -4.2968659, -42.1484909, 42.2323380
35: -40.9161873, 6.9673591, -40.9724922, 7.0281329, -43.7508850, 43.8358612
36: -44.4038239, 5.3223886, -44.4063339, 5.3856144, -45.4184418, 45.3843689
37: -59.3833160, 2.3848066, -59.4155731, 2.3491998, -55.0675812, 55.2013702
38: -50.8420563, 8.5865726, -50.8661346, 8.5827894, -59.4248466, 59.4527054
39: -52.0932541, 14.9951897, -52.1360321, 14.9037876, -66.9970398, 67.1312256
40: -47.8068161, 8.3605499, -47.8238983, 8.3098049, -53.0960083, 53.1467590
41: -31.8230247, 15.2404423, -31.8237572, 15.2369795, -45.4367371, 45.4471664
42: -27.1401138, 10.1549911, -27.1426086, 10.1614761, -36.7393723, 36.7475929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=176, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6893563, upper bound: 14.7690826
time: 48.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6640504, upper bound: 14.7690826
time: 41.26 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -23.9383087, 32.8784065, -23.9320068, 32.8656883, -54.3298187, 54.3489914
1: -7.7095718, 32.3290749, -7.6916485, 32.3106422, -36.5257797, 36.4898643
2: -4.7807169, 31.7844925, -4.7762432, 31.7561417, -33.0992432, 33.1736374
3: -8.9721851, 28.9130974, -8.9560375, 28.9072647, -32.4211960, 32.4570732
4: -10.1393604, 35.0682144, -10.1393929, 35.0415230, -43.3718719, 43.4252090
5: -11.1715651, 29.9287643, -11.1659355, 29.9205551, -38.1202164, 38.1894684
6: -38.7074165, 7.5073814, -38.6712685, 7.4838982, -44.4656296, 44.4406967
7: -15.3151817, 30.6265221, -15.3083010, 30.6089783, -41.6972046, 41.7203827
8: -15.5070667, 34.5678177, -15.4828310, 34.5376015, -47.0660095, 47.1140671
9: -10.3412428, 27.2037468, -10.3349209, 27.2056885, -35.8216629, 35.7128868
10: -28.4574547, 24.0243149, -28.4512119, 23.9642487, -50.5978394, 50.4200745
11: -35.7969322, 14.2796812, -35.7918701, 14.3076620, -49.4445038, 49.3557281
12: -49.4027252, 2.0419507, -49.3853569, 2.0426149, -44.0855331, 43.9897614
13: -28.8107643, 21.3660336, -28.8077183, 21.3409729, -49.5774384, 49.5986099
14: -71.0263901, -6.4150429, -70.9939575, -6.4028111, -64.6235809, 64.5789185
15: -17.4510746, 24.6832199, -17.4323921, 24.6828461, -42.1339188, 42.1156120
16: -27.5453186, 23.6723251, -27.5228310, 23.6801453, -48.5747147, 48.3900528
17: -71.2230377, -3.8741474, -71.1507034, -3.8869133, -67.3361206, 67.2765579
18: -34.7892303, 11.6586075, -34.7492981, 11.6670990, -40.6059456, 40.5688820
19: -25.7124596, 5.2713337, -25.7049065, 5.2685199, -29.9252014, 29.9231606
20: -26.4716454, 4.3867836, -26.4717369, 4.3765755, -29.1123199, 29.0559082
21: -31.3722687, 10.1311693, -31.3653259, 10.1553392, -40.4297791, 40.2892075
22: -33.6295090, 6.9675717, -33.6223297, 6.9813910, -38.4803314, 38.4518661
23: -26.9201927, 8.8951769, -26.9114017, 8.8930645, -35.2432785, 35.2384949
24: -23.2622986, 9.8584690, -23.2586441, 9.8477058, -32.7990456, 32.8316116
25: -29.2175674, 6.1130815, -29.2134609, 6.1012030, -34.3642654, 34.3932381
26: -42.9579430, 7.7739439, -42.9526634, 7.8375568, -43.7501221, 43.6386414
27: -26.6849651, 11.4740562, -26.6294003, 11.4800444, -37.7932739, 37.7486305
28: -29.6238289, 7.1716619, -29.5979118, 7.1639137, -36.6476135, 36.6345291
29: -32.6393967, 8.9975662, -32.6008301, 9.0138893, -41.6532860, 41.5983963
30: -37.5968704, 6.9714069, -37.5956802, 6.9753323, -44.5722046, 44.5670853
31: -31.4177952, 7.2874618, -31.4082146, 7.2700505, -37.5932617, 37.6825180
32: -33.6886864, 6.8345490, -33.6421623, 6.7699728, -40.4586601, 40.4767113
33: -43.9619560, 16.1041260, -43.9859085, 16.0216389, -57.1332092, 57.2868652
34: -50.7175140, -4.2054763, -50.7011261, -4.2479920, -42.2623978, 42.3115921
35: -40.9753723, 7.0780611, -40.9820023, 7.0767794, -43.8715973, 43.9156799
36: -44.4594803, 5.4482336, -44.4136200, 5.4409895, -45.5381165, 45.4912720
37: -59.4514160, 2.4514861, -59.4309921, 2.3780937, -55.1755219, 55.2315979
38: -50.9169121, 8.6916332, -50.8772469, 8.6280355, -59.5449486, 59.5688782
39: -52.1712723, 15.0189466, -52.1568413, 14.9134407, -67.0847168, 67.1757889
40: -47.8664627, 8.3936367, -47.8378983, 8.3235617, -53.1676331, 53.1889877
41: -31.8720322, 15.3153629, -31.8343163, 15.2689476, -45.5175018, 45.5247269
42: -27.1763573, 10.2435894, -27.1484318, 10.1993046, -36.8320312, 36.8319016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=176, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=225, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7342119, upper bound: 14.7690826
time: 33.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6640504, upper bound: 14.7690826
time: 40.26 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -23.8983002, 32.8761406, -23.9928970, 32.8686333, -54.3022003, 54.4030151
1: -7.6764011, 32.3285217, -7.7384758, 32.3184242, -36.5052109, 36.5283623
2: -4.7434902, 31.7893124, -4.8633928, 31.7860146, -33.1024780, 33.2547760
3: -8.9105015, 28.8978996, -8.9972038, 28.8862419, -32.3634109, 32.4764633
4: -10.1120186, 35.0735130, -10.2233820, 35.0745926, -43.3879166, 43.4999847
5: -11.1177330, 29.9175606, -11.2070990, 29.9124413, -38.0738373, 38.2173157
6: -38.7185402, 7.4693909, -38.7276649, 7.6121883, -44.6021347, 44.4553680
7: -15.2758579, 30.6267643, -15.3626719, 30.6234035, -41.6972733, 41.7451019
8: -15.4503031, 34.5666199, -15.5569162, 34.5633926, -47.0432281, 47.1751556
9: -10.3379574, 27.1866436, -10.3551149, 27.2285805, -35.8414993, 35.7077179
10: -28.4395180, 23.9303093, -28.4806976, 23.9995232, -50.6209717, 50.4499054
11: -35.7840233, 14.2852783, -35.8325005, 14.3430424, -49.4653625, 49.3946838
12: -49.3944511, 1.9508610, -49.3830719, 2.0710020, -44.0677948, 43.9469147
13: -28.8014088, 21.3198147, -28.8299561, 21.3483143, -49.5821686, 49.5663452
14: -70.9943161, -6.4035378, -71.1265106, -6.3541698, -64.6401443, 64.7229767
15: -17.4154129, 24.6760445, -17.4594173, 24.6806259, -42.0960388, 42.1354599
16: -27.5172310, 23.6628532, -27.5699043, 23.6895351, -48.5020599, 48.4302521
17: -71.1565323, -3.8666439, -71.3334656, -3.8363762, -67.3201599, 67.4668198
18: -34.7457352, 11.6485453, -34.7813759, 11.6776724, -40.5837746, 40.5894547
19: -25.6988716, 5.2607336, -25.7251625, 5.2903981, -29.9136200, 29.9373817
20: -26.4606342, 4.3567514, -26.4574223, 4.3951750, -29.0977135, 29.0473366
21: -31.3571682, 10.1288757, -31.3839664, 10.1771278, -40.4205551, 40.3055038
22: -33.6115417, 6.9631376, -33.6481552, 7.0016975, -38.4458160, 38.5091782
23: -26.9074478, 8.8893270, -26.9271622, 8.9142189, -35.2428055, 35.2521439
24: -23.2481365, 9.8586578, -23.2974243, 9.8790016, -32.8119278, 32.8715057
25: -29.2070980, 6.0980377, -29.2500401, 6.1283002, -34.3747025, 34.4285622
26: -42.9449730, 7.7635784, -42.9638214, 7.8428822, -43.7276917, 43.6596146
27: -26.6341228, 11.4617577, -26.6573563, 11.5163326, -37.8078995, 37.7665329
28: -29.5948372, 7.1602283, -29.6040878, 7.1906447, -36.6253128, 36.6324463
29: -32.6012154, 8.9876785, -32.6250076, 9.0201626, -41.6213760, 41.6126862
30: -37.5870552, 6.9636984, -37.6032257, 6.9984350, -44.5854912, 44.5669250
31: -31.4005356, 7.2859159, -31.4255066, 7.3089228, -37.5861130, 37.7113457
32: -33.6922112, 6.7533112, -33.6876030, 6.9000282, -40.5922394, 40.4409142
33: -43.9368095, 16.0191422, -43.9524879, 16.0595436, -57.1450653, 57.1902237
34: -50.7103119, -4.2474327, -50.7195473, -4.1602688, -42.3178864, 42.3413544
35: -40.9701958, 7.0774384, -40.9991150, 7.1216469, -43.8822174, 43.9567413
36: -44.4708862, 5.4327202, -44.4786072, 5.5610733, -45.6412506, 45.5643158
37: -59.4338646, 2.3740540, -59.4234505, 2.4560523, -55.2180176, 55.1736069
38: -50.9019623, 8.6173515, -50.9003601, 8.7279158, -59.6298790, 59.5177116
39: -52.1397820, 14.9018354, -52.1198044, 14.9557152, -67.0954971, 67.0216370
40: -47.8634186, 8.3231506, -47.8673019, 8.4391937, -53.2796326, 53.1525345
41: -31.8760090, 15.2638826, -31.8818016, 15.3964329, -45.6490250, 45.5276718
42: -27.1764450, 10.1809349, -27.1769638, 10.2923422, -36.9172287, 36.8049812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=176, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7489283, upper bound: 14.7690826
time: 35.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7198602, upper bound: 14.7690826
time: 20.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -23.8999920, 32.8837204, -24.0460243, 32.8976135, -54.3334885, 54.4594879
1: -7.6765251, 32.3340073, -7.7768593, 32.3414764, -36.5279236, 36.5734825
2: -4.7440805, 31.7950249, -4.9073372, 31.8102703, -33.1242523, 33.3052444
3: -8.9116669, 28.9116669, -9.0665340, 28.9274330, -32.3936462, 32.5540962
4: -10.1132574, 35.0779800, -10.2592163, 35.0971336, -43.4116516, 43.5423126
5: -11.1180410, 29.9288845, -11.2674942, 29.9471054, -38.1028290, 38.2886276
6: -38.7274704, 7.4704971, -38.7666740, 7.6562138, -44.6571884, 44.4945374
7: -15.2763309, 30.6335106, -15.4090519, 30.6481495, -41.7195282, 41.8054581
8: -15.4514103, 34.5728455, -15.6225758, 34.5977745, -47.0773773, 47.2490540
9: -10.3373985, 27.1897621, -10.3650036, 27.2556114, -35.8586731, 35.7453957
10: -28.4564171, 23.9357281, -28.5335541, 24.1160316, -50.7101898, 50.4926453
11: -35.7851028, 14.2854357, -35.8527451, 14.3511848, -49.4777832, 49.4207382
12: -49.4065781, 1.9547782, -49.4299011, 2.1755290, -44.1894455, 43.9904785
13: -28.8054752, 21.3263016, -28.8575745, 21.4086761, -49.6524811, 49.6029434
14: -70.9959564, -6.3956909, -71.1676178, -6.3214817, -64.6744766, 64.7719269
15: -17.4162674, 24.6814137, -17.5023785, 24.7034721, -42.1197395, 42.1837921
16: -27.5190105, 23.6659050, -27.6079483, 23.7163353, -48.5235138, 48.5319290
17: -71.1581345, -3.8446198, -71.4114304, -3.7632027, -67.3949280, 67.5668106
18: -34.7491264, 11.6519899, -34.8339615, 11.6958923, -40.6004562, 40.6437836
19: -25.7024441, 5.2607517, -25.7500668, 5.3053217, -29.9294662, 29.9729805
20: -26.4677734, 4.3578262, -26.4862404, 4.4298525, -29.1305008, 29.0746269
21: -31.3600464, 10.1291103, -31.4088001, 10.1835155, -40.4341202, 40.3366089
22: -33.6138573, 6.9632316, -33.6750565, 7.0135937, -38.4602509, 38.5554581
23: -26.9103394, 8.8895464, -26.9490318, 8.9252739, -35.2579956, 35.2781296
24: -23.2525215, 9.8598499, -23.3264084, 9.8881397, -32.8259163, 32.9014702
25: -29.2091026, 6.0998549, -29.2712460, 6.1539707, -34.3980789, 34.4561653
26: -42.9491882, 7.7647595, -42.9888840, 7.8592248, -43.7490158, 43.6916122
27: -26.6400566, 11.4711113, -26.7291737, 11.5478878, -37.8439331, 37.8415565
28: -29.5985870, 7.1632199, -29.6424942, 7.2170200, -36.6575012, 36.6939468
29: -32.6024780, 8.9932051, -32.6699753, 9.0459242, -41.6484032, 41.6631813
30: -37.5885315, 6.9637957, -37.6184769, 7.0142756, -44.6028061, 44.5822716
31: -31.4067554, 7.2857695, -31.4603004, 7.3220215, -37.6025314, 37.7494011
32: -33.7084045, 6.7548676, -33.7399750, 6.9872341, -40.6956406, 40.4948425
33: -43.9561729, 16.0205040, -44.0184402, 16.1503868, -57.2578888, 57.2558060
34: -50.7264442, -4.2474241, -50.7710571, -4.1146231, -42.3712082, 42.3815231
35: -40.9762344, 7.0767345, -41.0272141, 7.1265044, -43.9037552, 43.9831619
36: -44.4776344, 5.4323578, -44.5110855, 5.5805430, -45.6709442, 45.5971146
37: -59.4521675, 2.3747444, -59.4882431, 2.5395212, -55.3246002, 55.2366180
38: -50.9218826, 8.6176815, -50.9740982, 8.8082390, -59.7301216, 59.5917816
39: -52.1682472, 14.9030256, -52.2119598, 15.0784483, -67.2466965, 67.1149826
40: -47.8776093, 8.3232851, -47.9208832, 8.5136185, -53.3657990, 53.2029266
41: -31.8888817, 15.2646599, -31.9287872, 15.4540558, -45.7165604, 45.5704041
42: -27.1868134, 10.1828480, -27.2182350, 10.3617659, -36.9995346, 36.8428650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=176, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=225, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7632551, upper bound: 14.7690826
time: 16.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7632554, upper bound: 14.7690826
time: 34.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -24.0301361, 33.0302849, -23.9314213, 32.8658028, -54.4270782, 54.5055466
1: -7.7458658, 32.3897247, -7.6888695, 32.3100967, -36.5797119, 36.5376511
2: -4.8473692, 31.9178047, -4.7803869, 31.7553864, -33.1778107, 33.3039398
3: -9.0479841, 29.0923271, -8.9620705, 28.9061852, -32.5085144, 32.6212311
4: -10.1881609, 35.1769485, -10.1395454, 35.0406837, -43.4358749, 43.5253830
5: -11.2567253, 30.1218605, -11.1732101, 29.9199333, -38.2195892, 38.3739014
6: -38.7629585, 7.5280018, -38.6706734, 7.4788074, -44.5598602, 44.4638901
7: -15.3927526, 30.7275505, -15.3076258, 30.6074429, -41.8042908, 41.7945251
8: -15.5625191, 34.6901665, -15.4824543, 34.5367889, -47.1347656, 47.2386169
9: -10.4067402, 27.2414341, -10.3315754, 27.2040062, -35.9080505, 35.7276192
10: -28.6742783, 24.1104164, -28.4494743, 23.9707680, -50.8286743, 50.5007477
11: -36.0218124, 14.3329449, -35.7906456, 14.3120842, -49.6892242, 49.4073944
12: -49.6970024, 2.1734266, -49.3835182, 2.0514269, -44.3377228, 44.1620102
13: -28.8350182, 21.4631386, -28.8032227, 21.3417244, -49.6131897, 49.6742172
14: -71.2399216, -6.3378201, -70.9905090, -6.3937206, -64.8461990, 64.6526871
15: -17.4762020, 24.7793732, -17.4237614, 24.6825619, -42.1587639, 42.2031326
16: -27.6487446, 23.6851330, -27.5194721, 23.6763115, -48.6661758, 48.3801575
17: -71.3745270, -3.7884102, -71.1487732, -3.8774929, -67.4970322, 67.3603668
18: -34.9554405, 11.6906261, -34.7486534, 11.6666317, -40.7881126, 40.6115913
19: -25.8203907, 5.2927837, -25.7046566, 5.2701273, -30.0102158, 29.9463654
20: -26.5826702, 4.4189587, -26.4719238, 4.3783684, -29.2190552, 29.1113281
21: -31.5481453, 10.1743431, -31.3645821, 10.1597176, -40.6124039, 40.3437271
22: -33.7057724, 7.0140505, -33.6204185, 6.9778509, -38.5466843, 38.5109787
23: -27.0300312, 8.9214296, -26.9108181, 8.8933144, -35.3365707, 35.2750244
24: -23.3237381, 9.8742056, -23.2578220, 9.8450165, -32.8486671, 32.8519821
25: -29.2871380, 6.1481180, -29.2123871, 6.1017241, -34.4201202, 34.4315186
26: -43.2418404, 7.8904452, -42.9509583, 7.8468795, -44.0325775, 43.7579117
27: -26.7612495, 11.4922562, -26.6295052, 11.4785614, -37.8650894, 37.7655716
28: -29.6907787, 7.1994057, -29.5977612, 7.1616902, -36.7006454, 36.6777115
29: -32.7169609, 9.0365295, -32.5989838, 9.0146122, -41.7315750, 41.6355133
30: -37.7158051, 7.0129814, -37.5950127, 6.9762630, -44.6920700, 44.6079941
31: -31.5534077, 7.3041320, -31.4080124, 7.2703342, -37.6777077, 37.7285538
32: -33.7695351, 6.8647842, -33.6413193, 6.7679005, -40.5374374, 40.5061035
33: -44.0170288, 16.2185974, -43.9890785, 16.0203476, -57.1701202, 57.4353333
34: -50.7502975, -4.1210709, -50.7027893, -4.2499781, -42.2478790, 42.4411392
35: -41.0134277, 7.1809459, -40.9841423, 7.0751367, -43.8660660, 44.0605621
36: -44.4769592, 5.4889836, -44.4125595, 5.4371767, -45.5272369, 45.5641632
37: -59.5283279, 2.4845715, -59.4297523, 2.3728027, -55.2124939, 55.3153534
38: -50.9636345, 8.7347660, -50.8760948, 8.6247149, -59.5883484, 59.6108627
39: -52.2237778, 15.0706015, -52.1561241, 14.9088907, -67.1326675, 67.2267227
40: -47.9196320, 8.4336061, -47.8377380, 8.3212347, -53.2180405, 53.2350388
41: -31.9278240, 15.3399315, -31.8337345, 15.2641897, -45.5811691, 45.5508347
42: -27.2621574, 10.2755413, -27.1480541, 10.1932487, -36.9603043, 36.8822937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=225, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7089043, upper bound: 14.7690825
time: 18.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7400418, upper bound: 14.7690825
time: 40.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -23.8860970, 33.0051842, -23.7893181, 32.8295441, -54.2495880, 54.3424759
1: -7.6473694, 32.3722191, -7.6005793, 32.2900772, -36.4513779, 36.4435654
2: -4.7468166, 31.9061985, -4.7394686, 31.7607555, -33.0743942, 33.2682571
3: -8.9371338, 29.0490799, -8.9087563, 28.8748188, -32.3612747, 32.5624161
4: -10.0766392, 35.1664200, -10.0524807, 35.0375023, -43.3162079, 43.4397430
5: -11.1480293, 30.0857430, -11.1037979, 29.8853645, -38.0662994, 38.3008423
6: -38.7449532, 7.4422960, -38.7056770, 7.5112095, -44.5852203, 44.3991699
7: -15.2662392, 30.7117386, -15.1808920, 30.5872173, -41.6372604, 41.6883621
8: -15.4214697, 34.6654358, -15.3845654, 34.5250244, -46.9690857, 47.1290283
9: -10.3333426, 27.2122555, -10.2210255, 27.1899261, -35.7940521, 35.6222954
10: -28.5780830, 23.9958820, -28.4064598, 23.9605198, -50.7365112, 50.4367065
11: -35.9892311, 14.3236275, -35.8029289, 14.3113880, -49.6462097, 49.4059448
12: -49.6551590, 1.9969521, -49.3575172, 1.9174447, -44.2294540, 43.9427338
13: -28.7946568, 21.3908882, -28.8024788, 21.3298531, -49.5324554, 49.6050720
14: -71.1391602, -6.3530846, -70.9903641, -6.3806381, -64.7585220, 64.6372833
15: -17.4039574, 24.7535820, -17.3820763, 24.6617680, -42.0657272, 42.1356583
16: -27.5433502, 23.6561584, -27.4176865, 23.6488762, -48.4700775, 48.2754669
17: -71.2807922, -3.8281174, -71.2884979, -3.8205891, -67.4602051, 67.4603806
18: -34.8965912, 11.6587362, -34.7631302, 11.6564789, -40.7128296, 40.6067505
19: -25.7866726, 5.2751880, -25.7003593, 5.2791986, -29.9728165, 29.9110909
20: -26.5471363, 4.3777723, -26.4496441, 4.3840351, -29.1966362, 29.0279465
21: -31.5092659, 10.1632652, -31.3476353, 10.1642933, -40.5785141, 40.2938995
22: -33.6691132, 6.9791446, -33.6042252, 6.9431028, -38.4664612, 38.4772339
23: -27.0016613, 8.8976250, -26.9052124, 8.8809118, -35.2899857, 35.2314072
24: -23.2868042, 9.8615971, -23.2849541, 9.8621464, -32.8273392, 32.8602333
25: -29.2572651, 6.0931301, -29.2171021, 6.0570178, -34.3454819, 34.3895531
26: -43.2090645, 7.8526616, -42.9275703, 7.8110685, -43.9781570, 43.7009201
27: -26.6875591, 11.4505749, -26.6385612, 11.4964247, -37.8307953, 37.7330856
28: -29.6456356, 7.1452198, -29.5752182, 7.1324639, -36.6108627, 36.5971909
29: -32.6624298, 8.9874935, -32.5881119, 8.9738407, -41.6362686, 41.5756073
30: -37.6917648, 6.9861679, -37.5852127, 6.9681091, -44.6598740, 44.5713806
31: -31.5094318, 7.2885246, -31.3948536, 7.2803040, -37.6338654, 37.6634598
32: -33.7372551, 6.7381153, -33.6725616, 6.8123007, -40.5495567, 40.4106750
33: -43.9360046, 16.0892677, -43.9169159, 15.9725780, -57.0726776, 57.2371063
34: -50.7090836, -4.2148228, -50.6977081, -4.2710500, -42.2446289, 42.3365936
35: -40.9844971, 7.1281424, -40.9563446, 7.0128064, -43.8234482, 43.9542770
36: -44.4650116, 5.4149089, -44.4379463, 5.4345670, -45.5385742, 45.5024948
37: -59.4590721, 2.3738823, -59.3902817, 2.3884830, -55.2238007, 55.1504974
38: -50.8995819, 8.6105900, -50.8623276, 8.6242247, -59.5238075, 59.4729156
39: -52.1185341, 14.9390945, -52.0939713, 14.9320707, -67.0506058, 67.0330658
40: -47.8740807, 8.3467627, -47.8361053, 8.4064445, -53.2658386, 53.1489182
41: -31.8955097, 15.2514009, -31.8581524, 15.3217697, -45.6204376, 45.4879990
42: -27.2328014, 10.1697187, -27.1639786, 10.2030411, -36.9558411, 36.7680092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=223, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109636, upper bound: 14.7099182
time: 35.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109640, upper bound: 14.7099182
time: 36.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -23.9392586, 33.0342102, -23.7910194, 32.8371620, -54.3060989, 54.3737793
1: -7.6857891, 32.3952866, -7.6006985, 32.2955742, -36.4964981, 36.4663277
2: -4.7907543, 31.9304008, -4.7401133, 31.7664738, -33.1248322, 33.2899246
3: -9.0064278, 29.0902557, -8.9099598, 28.8885288, -32.4388046, 32.5926056
4: -10.1124763, 35.1889114, -10.0536966, 35.0419846, -43.3585587, 43.4634094
5: -11.2083483, 30.1204147, -11.1041069, 29.8966560, -38.1375351, 38.3298187
6: -38.7838745, 7.4863491, -38.7145920, 7.5123234, -44.6243591, 44.4542313
7: -15.3127012, 30.7364082, -15.1814175, 30.5939465, -41.6975403, 41.7105865
8: -15.4871025, 34.6997414, -15.3857126, 34.5312958, -47.0430298, 47.1631470
9: -10.3432512, 27.2393055, -10.2204542, 27.1930695, -35.8317642, 35.6394882
10: -28.6309166, 24.1123943, -28.4233246, 23.9659519, -50.7791824, 50.5259705
11: -36.0094528, 14.3317919, -35.8040009, 14.3116016, -49.6722336, 49.4182663
12: -49.7020149, 2.1014943, -49.3696747, 1.9213319, -44.2730408, 44.0644379
13: -28.8222904, 21.4511909, -28.8065357, 21.3362617, -49.5691376, 49.6753006
14: -71.1801910, -6.3204346, -70.9920425, -6.3727894, -64.8074036, 64.6716080
15: -17.4469547, 24.7764359, -17.3829384, 24.6671143, -42.1140671, 42.1593742
16: -27.5813656, 23.6829224, -27.4194736, 23.6518364, -48.5717239, 48.2969437
17: -71.3587494, -3.7548714, -71.2900696, -3.7986069, -67.5601425, 67.5352020
18: -34.9491653, 11.6769485, -34.7664528, 11.6599331, -40.7672195, 40.6234131
19: -25.8115635, 5.2901220, -25.7039299, 5.2792101, -30.0083542, 29.9269218
20: -26.5759621, 4.4125123, -26.4567642, 4.3851070, -29.2239456, 29.0607796
21: -31.5340385, 10.1696520, -31.3504810, 10.1645250, -40.6095734, 40.3074875
22: -33.6959572, 6.9909945, -33.6065674, 6.9432745, -38.5126877, 38.4916687
23: -27.0234795, 8.9086475, -26.9081326, 8.8811493, -35.3159485, 35.2466469
24: -23.3157768, 9.8707705, -23.2893734, 9.8633442, -32.8573265, 32.8742294
25: -29.2784328, 6.1188531, -29.2190628, 6.0588579, -34.3730850, 34.4129639
26: -43.2341232, 7.8690681, -42.9317780, 7.8122201, -44.0101166, 43.7223434
27: -26.7593994, 11.4821053, -26.6444740, 11.5057621, -37.9058151, 37.7691498
28: -29.6839523, 7.1716623, -29.5790100, 7.1354814, -36.6722565, 36.6294403
29: -32.7073669, 9.0133114, -32.5893860, 8.9793491, -41.6867142, 41.6026993
30: -37.7069778, 7.0020180, -37.5866356, 6.9682245, -44.6752014, 44.5886536
31: -31.5442123, 7.3016348, -31.4010658, 7.2801347, -37.6719131, 37.6798134
32: -33.7895470, 6.8253517, -33.6887779, 6.8138714, -40.6034164, 40.5141296
33: -44.0019760, 16.1801376, -43.9363098, 15.9738979, -57.1382446, 57.3498917
34: -50.7605209, -4.1691790, -50.7138214, -4.2710648, -42.2847366, 42.3899078
35: -41.0125198, 7.1329803, -40.9623222, 7.0121207, -43.8498230, 43.9758530
36: -44.4973869, 5.4343147, -44.4447327, 5.4342575, -45.5711823, 45.5322266
37: -59.5238762, 2.4573274, -59.4085922, 2.3891521, -55.2868500, 55.2570648
38: -50.9732590, 8.6909084, -50.8822212, 8.6245575, -59.5978165, 59.5731277
39: -52.2107086, 15.0618973, -52.1224365, 14.9332609, -67.1439667, 67.1843338
40: -47.9276123, 8.4211836, -47.8503189, 8.4065933, -53.3162537, 53.2349854
41: -31.9423256, 15.3090038, -31.8710136, 15.3225107, -45.6630707, 45.5555344
42: -27.2739906, 10.2391586, -27.1743679, 10.2049370, -36.9937439, 36.8503036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=225, inp2_unstable=223, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109636, upper bound: 14.7242453
time: 36.32 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109640, upper bound: 14.7242453
time: 46.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -23.9791870, 33.0085373, -24.0032997, 32.8882332, -54.4039764, 54.5549545
1: -7.7084589, 32.3747406, -7.7399788, 32.3322525, -36.5595779, 36.5756989
2: -4.8053327, 31.9094963, -4.8723831, 31.7983971, -33.1754990, 33.3907013
3: -8.9805565, 29.0534859, -9.0089645, 28.9088364, -32.4468231, 32.6444931
4: -10.1539602, 35.1697922, -10.2304087, 35.0861740, -43.4441071, 43.6079636
5: -11.1982899, 30.0914021, -11.2190723, 29.9310703, -38.1706390, 38.4042740
6: -38.7515717, 7.4853325, -38.7495651, 7.6117706, -44.6819000, 44.4930878
7: -15.3479404, 30.7140732, -15.3674641, 30.6355591, -41.7794113, 41.8442993
8: -15.4988327, 34.6710281, -15.5633926, 34.5805473, -47.1095734, 47.3020935
9: -10.3942604, 27.2162933, -10.3609743, 27.2349358, -35.9044266, 35.7459259
10: -28.6235199, 24.0052319, -28.5117836, 24.0171509, -50.8394928, 50.5427933
11: -36.0029907, 14.3343153, -35.8372345, 14.3516779, -49.7053223, 49.4511261
12: -49.6600494, 2.0707340, -49.4099083, 2.0914335, -44.3548050, 44.0843201
13: -28.8104172, 21.4031639, -28.8406124, 21.3628578, -49.6041183, 49.6557617
14: -71.2009277, -6.3450985, -71.1300049, -6.3262043, -64.8747253, 64.7849045
15: -17.4346390, 24.7587128, -17.4566288, 24.6938686, -42.1285095, 42.2153397
16: -27.6119175, 23.6614380, -27.5753136, 23.6997871, -48.5933075, 48.4205551
17: -71.3014221, -3.8215218, -71.3380661, -3.7863693, -67.5150528, 67.5165405
18: -34.9044533, 11.6708794, -34.7881775, 11.6868763, -40.7642365, 40.6338921
19: -25.7970695, 5.2809682, -25.7345734, 5.2931786, -29.9979401, 29.9613457
20: -26.5539703, 4.3854976, -26.4752769, 4.4003825, -29.2135925, 29.0935593
21: -31.5242500, 10.1707115, -31.3921089, 10.1828938, -40.6052399, 40.3579102
22: -33.6808701, 7.0049710, -33.6531677, 7.0028448, -38.5263519, 38.5540619
23: -27.0094547, 8.9126835, -26.9344864, 8.9173584, -35.3369980, 35.2877808
24: -23.2959137, 9.8707142, -23.3101768, 9.8799858, -32.8548355, 32.8986053
25: -29.2663231, 6.1271915, -29.2593174, 6.1347055, -34.4294815, 34.4680023
26: -43.2184601, 7.8734307, -42.9725037, 7.8588324, -44.0212402, 43.7676620
27: -26.6981621, 11.4615784, -26.6697330, 11.5332146, -37.8831406, 37.7799530
28: -29.6531830, 7.1751127, -29.6125603, 7.2011681, -36.6877213, 36.6662827
29: -32.6730576, 9.0136309, -32.6289101, 9.0339050, -41.7069626, 41.6425400
30: -37.7004852, 6.9999046, -37.6081085, 7.0047903, -44.7052765, 44.6080132
31: -31.5209503, 7.3009629, -31.4404640, 7.3108788, -37.6762772, 37.7516785
32: -33.7444496, 6.7783155, -33.7153854, 6.9032373, -40.6476860, 40.4937019
33: -43.9536552, 16.1290646, -43.9938774, 16.0628014, -57.1653137, 57.3553848
34: -50.7157822, -4.1659288, -50.7485580, -4.1593285, -42.3239059, 42.4504395
35: -40.9939651, 7.1768150, -41.0155029, 7.1235185, -43.9033356, 44.0749741
36: -44.4722900, 5.4702539, -44.4935951, 5.5604062, -45.6453857, 45.6221695
37: -59.4745216, 2.4027767, -59.4584312, 2.4552064, -55.2540894, 55.2582932
38: -50.9106750, 8.6558323, -50.9371300, 8.7292709, -59.6399460, 59.5929642
39: -52.1393509, 14.9487658, -52.1720161, 14.9558849, -67.0952377, 67.1207809
40: -47.8881340, 8.3604994, -47.8957291, 8.4395227, -53.3080673, 53.2205582
41: -31.9060268, 15.2834120, -31.9070778, 15.3966942, -45.6980133, 45.5685883
42: -27.2386093, 10.2075462, -27.2001934, 10.2916241, -37.0402069, 36.8606148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1352
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109636, upper bound: 14.7547540
time: 31.17 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109640, upper bound: 14.7547540
time: 35.19 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -24.0323429, 33.0375557, -24.0049915, 32.8958702, -54.4604187, 54.5862885
1: -7.7468443, 32.3978271, -7.7400723, 32.3377838, -36.6047058, 36.5985146
2: -4.8492751, 31.9336624, -4.8730025, 31.8041039, -33.2259674, 33.4124146
3: -9.0498810, 29.0946674, -9.0101280, 28.9226036, -32.5243607, 32.6747055
4: -10.1898022, 35.1923027, -10.2316093, 35.0906219, -43.4863892, 43.6316299
5: -11.2586365, 30.1260910, -11.2193556, 29.9423752, -38.2419128, 38.4332962
6: -38.7905388, 7.5293560, -38.7585106, 7.6128902, -44.7209778, 44.5481186
7: -15.3943481, 30.7387352, -15.3680048, 30.6422958, -41.8396912, 41.8665390
8: -15.5644741, 34.7053528, -15.5645294, 34.5867844, -47.1834869, 47.3362045
9: -10.4041586, 27.2433395, -10.3603992, 27.2380733, -35.9421005, 35.7631378
10: -28.6763458, 24.1217823, -28.5286827, 24.0225639, -50.8821716, 50.6320953
11: -36.0232010, 14.3424511, -35.8382607, 14.3518343, -49.7313843, 49.4635086
12: -49.7068634, 2.1752706, -49.4219894, 2.0953250, -44.3983459, 44.2060394
13: -28.8380184, 21.4635277, -28.8446579, 21.3692360, -49.6408234, 49.7260742
14: -71.2420044, -6.3124256, -71.1316910, -6.3183594, -64.9236450, 64.8192673
15: -17.4776077, 24.7815418, -17.4575214, 24.6992054, -42.1768112, 42.2390633
16: -27.6499195, 23.6882782, -27.5770950, 23.7028198, -48.6949463, 48.4420471
17: -71.3793716, -3.7482948, -71.3396378, -3.7643242, -67.6150513, 67.5913391
18: -34.9570694, 11.6891146, -34.7915306, 11.6903076, -40.8186226, 40.6505127
19: -25.8219681, 5.2959156, -25.7381477, 5.2932138, -30.0334930, 29.9771805
20: -26.5828323, 4.4202051, -26.4823837, 4.4014764, -29.2408600, 29.1263809
21: -31.5490284, 10.1770840, -31.3949432, 10.1831017, -40.6363068, 40.3715057
22: -33.7077255, 7.0168319, -33.6554947, 7.0029769, -38.5725784, 38.5685425
23: -27.0312843, 8.9237289, -26.9373913, 8.9175701, -35.3629761, 35.3030167
24: -23.3248978, 9.8798666, -23.3145714, 9.8811550, -32.8847809, 32.9125938
25: -29.2874928, 6.1528878, -29.2613106, 6.1365256, -34.4570312, 34.4913635
26: -43.2435989, 7.8898020, -42.9767075, 7.8600526, -44.0532532, 43.7890472
27: -26.7700157, 11.4931316, -26.6756439, 11.5425262, -37.9582214, 37.8159943
28: -29.6915016, 7.2015581, -29.6163368, 7.2041612, -36.7492065, 36.6985321
29: -32.7180214, 9.0394535, -32.6301804, 9.0394249, -41.7574463, 41.6696320
30: -37.7157288, 7.0157194, -37.6095390, 7.0048437, -44.7205734, 44.6252594
31: -31.5557823, 7.3140450, -31.4466839, 7.3107491, -37.7143745, 37.7680435
32: -33.7968254, 6.8655748, -33.7316093, 6.9047709, -40.7015953, 40.5971832
33: -44.0196075, 16.2199287, -44.0132256, 16.0641594, -57.2308655, 57.4682007
34: -50.7672653, -4.1202927, -50.7646790, -4.1593370, -42.3639984, 42.5037537
35: -41.0220642, 7.1816812, -41.0215492, 7.1228518, -43.9297028, 44.0965195
36: -44.5046616, 5.4897547, -44.5003357, 5.5601106, -45.6780701, 45.6518784
37: -59.5393181, 2.4862332, -59.4767609, 2.4559021, -55.3171310, 55.3648376
38: -50.9843674, 8.7361555, -50.9570198, 8.7295675, -59.7139359, 59.6931763
39: -52.2315140, 15.0715790, -52.2005196, 14.9570532, -67.1885681, 67.2720947
40: -47.9416275, 8.4349699, -47.9098625, 8.4396830, -53.3584442, 53.3066177
41: -31.9529171, 15.3410444, -31.9199505, 15.3974524, -45.7406387, 45.6361618
42: -27.2798347, 10.2769556, -27.2105675, 10.2935266, -37.0780945, 36.9429398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=225, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1367
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1489
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 895

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109636, upper bound: 14.7690825
time: 36.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109640, upper bound: 14.7690825
time: 40.88 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 79.65 seconds
IS_A1_A1_B2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6429702, upper bound: 14.7877928
IS_A1_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6429702, upper bound: 14.8109642
IS_A1_A1_B2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6429702, upper bound: 14.7877928
IS_A1_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6429702, upper bound: 14.8109642
IS_A1_A1_B2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6878397, upper bound: 14.7661502
IS_A1_A1_B2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6429702, upper bound: 14.7662030
IS_A1_A1_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7021987, upper bound: 14.7661502
IS_A1_A1_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6573261, upper bound: 14.7662029
IS_A1_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6720688, upper bound: 14.8109638
IS_A1_A1_B2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6720688, upper bound: 14.7662029
IS_A1_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6864035, upper bound: 14.8109638
IS_A1_A1_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6864035, upper bound: 14.7662029
IS_A1_A2_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6720689, upper bound: 14.7877927
IS_A1_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6720689, upper bound: 14.8109641
IS_A1_A2_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7242453, upper bound: 14.7877927
IS_A1_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7242453, upper bound: 14.8109641
IS_A1_A2_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7547536, upper bound: 14.7661501
IS_A1_A2_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6720689, upper bound: 14.7662028
IS_A1_A2_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7690822, upper bound: 14.7661501
IS_A1_A2_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7242453, upper bound: 14.7662028
IS_A2_B2_A1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6893563, upper bound: 14.7690826
IS_A2_B2_A1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6640504, upper bound: 14.7690826
IS_A2_B2_A1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7342119, upper bound: 14.7690826
IS_A2_B2_A1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.6640504, upper bound: 14.7690826
IS_A2_B2_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7489283, upper bound: 14.7690826
IS_A2_B2_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7198602, upper bound: 14.7690826
IS_A2_B2_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7632551, upper bound: 14.7690826
IS_A2_B2_A1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7632554, upper bound: 14.7690826
IS_A2_B2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7089043, upper bound: 14.7690825
IS_A2_B2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.7400418, upper bound: 14.7690825
IS_A2_B2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.8109636, upper bound: 14.7099182
IS_A2_B2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.8109640, upper bound: 14.7099182
IS_A2_B2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.8109636, upper bound: 14.7242453
IS_A2_B2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.8109640, upper bound: 14.7242453
IS_A2_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.8109636, upper bound: 14.7547540
IS_A2_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.8109640, upper bound: 14.7547540
IS_A2_B2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.8109636, upper bound: 14.7690825
IS_A2_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 79.65
Output dim: 2, lower bound: -14.8109640, upper bound: 14.7690825

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 49.75 + 3578.26 = 3628.00 seconds
