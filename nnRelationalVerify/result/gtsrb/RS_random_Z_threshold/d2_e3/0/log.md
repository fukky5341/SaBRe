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
execution time: IAR + RelationalAnalysis = 2.73 + 46.52 = 49.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -14.8236368, upper bound: 14.8236369

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 766

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8167145, upper bound: 14.7932057
time: 35.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7932056, upper bound: 14.8167146
time: 39.91 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 75.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 75.50
Output dim: 2, lower bound: -14.8167145, upper bound: 14.7932057
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 75.50
Output dim: 2, lower bound: -14.7932056, upper bound: 14.8167146

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4416351, 54.4370880
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5710907, 36.5663757
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2452927, 33.2421265
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5341797, 32.5319901
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4997406, 43.4988556
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2653503, 38.2630615
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5371094, 44.5374146
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8074570, 41.7989273
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1964417, 47.1923828
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8685608, 35.8656273
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6878128, 50.6884079
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5296249, 49.5218201
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2050095, 44.2158585
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6169052, 49.6210785
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5966263, 48.5848846
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6372108, 40.6375732
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9794846, 29.9739304
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1456642, 29.1456528
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4730377, 40.4658966
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5473938, 38.5548935
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2861481, 35.2825699
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8607826, 32.8592224
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4306030, 34.4298859
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8304825, 43.8388214
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8181305, 37.8168526
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6652756, 36.6642227
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6758156, 37.6681938
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2853546, 57.2883148
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3972092, 42.4077072
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -44.0070496, 44.0119324
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6169739, 45.6248932
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2762070, 55.2823105
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2274780, 53.2306061
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5971527, 45.5973129
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8910294, 36.8901825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1648

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 978

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8164493, upper bound: 14.7533887
time: 24.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7768995, upper bound: 14.7929400
time: 40.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4370880, 54.4408264
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5663834, 36.5702744
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2421265, 33.2446671
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5319977, 32.5336609
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4988556, 43.4993515
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2630615, 38.2652054
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5373840, 44.5371094
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7989273, 41.8042145
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1923828, 47.1943817
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8656311, 35.8660660
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6884079, 50.6891479
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5218124, 49.5255661
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2132187, 44.2050095
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6185379, 49.6169052
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5848770, 48.5910416
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6379433, 40.6372070
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9739304, 29.9775314
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1456490, 29.1457558
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4658966, 40.4689713
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5501099, 38.5473862
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2825699, 35.2848015
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8592186, 32.8603897
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4298859, 34.4305496
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8384476, 43.8304825
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8168564, 37.8176308
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6642227, 36.6648712
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6681938, 37.6741982
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2882080, 57.2853546
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.4060440, 42.3972015
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -44.0117950, 44.0070496
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6235657, 45.6169739
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2817917, 55.2762146
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2302246, 53.2274857
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5972900, 45.5971451
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8901825, 36.8905182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 769

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1587

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7916941, upper bound: 14.8091190
time: 33.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7856038, upper bound: 14.8152037
time: 49.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 84.75 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 84.75
Output dim: 2, lower bound: -14.8164493, upper bound: 14.7533887
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 84.75
Output dim: 2, lower bound: -14.7768995, upper bound: 14.7929400
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 84.75
Output dim: 2, lower bound: -14.7916941, upper bound: 14.8091190
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 84.75
Output dim: 2, lower bound: -14.7856038, upper bound: 14.8152037

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4407349, 54.4356537
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5691757, 36.5633011
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2438660, 33.2400208
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5314255, 32.5283585
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4979858, 43.4960175
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2640686, 38.2615204
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5354462, 44.5362625
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8049164, 41.7948151
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1949005, 47.1898804
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8679810, 35.8648643
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6876144, 50.6884842
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5298386, 49.5218048
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1996002, 44.2124710
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6156235, 49.6211395
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5960541, 48.5840073
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6373940, 40.6375465
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9789886, 29.9733620
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1456451, 29.1456375
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4734497, 40.4658585
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5455017, 38.5537262
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2862244, 35.2825699
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8612137, 32.8591843
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4303207, 34.4295807
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8280563, 43.8372955
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8185959, 37.8167191
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6652985, 36.6642227
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6762047, 37.6681595
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2845459, 57.2878265
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3957214, 42.4068069
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -44.0045929, 44.0102921
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6128235, 45.6222992
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2692490, 55.2779770
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2258911, 53.2299500
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5953217, 45.5961685
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8903580, 36.8896027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1371

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1728

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8147470, upper bound: 14.7528539
time: 46.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8159257, upper bound: 14.7516702
time: 39.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4139633, 54.4134750
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5559082, 36.5575638
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2342606, 33.2353745
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5330124, 32.5346909
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4974213, 43.4998856
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2683334, 38.2696304
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5372772, 44.5369186
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7990112, 41.8042603
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1845322, 47.1847458
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8791199, 35.8835106
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6870422, 50.6934586
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5282822, 49.5301132
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2152710, 44.2071533
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5990524, 49.5938873
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5829620, 48.5870819
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6197510, 40.6217079
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9794693, 29.9816360
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1147728, 29.1198578
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4626923, 40.4660950
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5291748, 38.5297394
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2830048, 35.2853127
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8507462, 32.8532333
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4255829, 34.4264984
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8067932, 43.8032837
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8021240, 37.8050995
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6593094, 36.6604538
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6672401, 37.6732635
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2882843, 57.2854156
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.4002686, 42.3935928
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9955750, 43.9870682
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6166077, 45.6088181
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2457504, 55.2333145
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2315216, 53.2287750
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5896988, 45.5884094
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8906403, 36.8909760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1368

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7866091, upper bound: 14.8079657
time: 38.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7905391, upper bound: 14.8040364
time: 33.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4097214, 54.4177094
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5536880, 36.5597763
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2328415, 33.2367935
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5330276, 32.5346756
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4994049, 43.4978943
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2674942, 38.2704620
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5371857, 44.5370102
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7989655, 41.8042984
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1827927, 47.1864853
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8830719, 35.8795624
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6927185, 50.6877747
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5263596, 49.5320282
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2153625, 44.2070618
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5955124, 49.5974121
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5809479, 48.5891113
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6224365, 40.6190300
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9780350, 29.9830780
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1197548, 29.1148758
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4630280, 40.4657822
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5324402, 38.5264587
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2830811, 35.2852402
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8520737, 32.8519058
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4258347, 34.4262505
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8112183, 43.7988510
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8043289, 37.8028946
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6598129, 36.6599503
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6672554, 37.6732445
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2882690, 57.2854385
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.4024353, 42.3914185
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9918365, 43.9907990
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6154022, 45.6100082
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2388840, 55.2401657
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2314911, 53.2288055
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5885696, 45.5895462
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8906403, 36.8909760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1354

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7851849, upper bound: 14.8150595
time: 42.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7854587, upper bound: 14.8147855
time: 34.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 79.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 79.30
Output dim: 2, lower bound: -14.8147470, upper bound: 14.7528539
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 79.30
Output dim: 2, lower bound: -14.8159257, upper bound: 14.7516702
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 79.30
Output dim: 2, lower bound: -14.7866091, upper bound: 14.8079657
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 79.30
Output dim: 2, lower bound: -14.7905391, upper bound: 14.8040364
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 79.30
Output dim: 2, lower bound: -14.7851849, upper bound: 14.8150595
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 79.30
Output dim: 2, lower bound: -14.7854587, upper bound: 14.8147855

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4409027, 54.4356079
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5692902, 36.5628204
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2438812, 33.2397308
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5320206, 32.5278320
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4981384, 43.4957581
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2642822, 38.2613449
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5354004, 44.5363922
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8048172, 41.7943649
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1953430, 47.1898193
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8665924, 35.8639183
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6838150, 50.6857376
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5297775, 49.5221329
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1994705, 44.2127838
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6155777, 49.6211319
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5944061, 48.5829544
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6371651, 40.6375160
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9777985, 29.9727173
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1448288, 29.1454506
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4729156, 40.4656830
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5449982, 38.5532761
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2854996, 35.2822151
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8612061, 32.8591995
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4294662, 34.4291306
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8272781, 43.8370438
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8183975, 37.8166122
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6648636, 36.6640091
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6757545, 37.6682320
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2836609, 57.2864304
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3951721, 42.4058685
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -44.0032501, 44.0085220
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6127014, 45.6225739
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2686081, 55.2778702
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2255554, 53.2298584
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5952911, 45.5963287
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8902588, 36.8897018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1636

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8138652, upper bound: 14.7278167
time: 43.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7897050, upper bound: 14.7519687
time: 41.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4406891, 54.4358215
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5687027, 36.5634079
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2435760, 33.2400360
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5309067, 32.5289459
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4977264, 43.4961700
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2638855, 38.2617416
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5355835, 44.5362091
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8044510, 41.7947235
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1948242, 47.1903305
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8670349, 35.8634796
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6848679, 50.6846771
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5301590, 49.5217590
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1999130, 44.2123413
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6156082, 49.6211014
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5949860, 48.5823669
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6373634, 40.6373177
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9783478, 29.9721680
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1454544, 29.1448250
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4732819, 40.4653168
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5450592, 38.5532150
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2858658, 35.2818451
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8612366, 32.8591690
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4298706, 34.4287338
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8277969, 43.8365402
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8184891, 37.8165207
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6650772, 36.6637955
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6762810, 37.6677055
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2831421, 57.2869415
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3947906, 42.4062576
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -44.0028229, 44.0089417
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6130981, 45.6221848
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2691574, 55.2773438
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2257996, 53.2296371
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5954742, 45.5961456
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8904572, 36.8895035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 774

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8156554, upper bound: 14.7368414
time: 42.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8011035, upper bound: 14.7513966
time: 38.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4080048, 54.4157028
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5531311, 36.5587044
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2354736, 33.2386475
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5349426, 32.5360260
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4949341, 43.4927826
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2676926, 38.2702866
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5331573, 44.5334625
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8010712, 41.8059540
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1792450, 47.1823959
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8839035, 35.8808556
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6828003, 50.6795731
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5224686, 49.5286865
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2118225, 44.2041626
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5930023, 49.5951080
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5853424, 48.5945740
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6162834, 40.6118813
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9785461, 29.9841805
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1194611, 29.1145477
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4618683, 40.4649124
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5315628, 38.5254440
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2834015, 35.2857285
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8508224, 32.8504219
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4266510, 34.4273567
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8064651, 43.7930984
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8001175, 37.7978897
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6604156, 36.6607056
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6657791, 37.6721878
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2830505, 57.2810364
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.4018326, 42.3917694
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9868698, 43.9863586
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6157303, 45.6103210
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2315445, 55.2340164
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2280502, 53.2264099
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5842743, 45.5859146
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8869705, 36.8878517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1669

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7850964, upper bound: 14.8082956
time: 35.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7779724, upper bound: 14.8149862
time: 37.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4077301, 54.4159698
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5525970, 36.5592308
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2346954, 33.2394257
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5343628, 32.5365906
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4942932, 43.4934387
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2672958, 38.2706604
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5336456, 44.5329666
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8006134, 41.8063965
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1786957, 47.1829605
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8843765, 35.8803825
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6845245, 50.6778488
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5230179, 49.5281372
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2124634, 44.2035217
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5932159, 49.5948868
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5863953, 48.5935211
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6152840, 40.6128769
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9791336, 29.9835892
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1194305, 29.1145744
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4621582, 40.4646378
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5314407, 38.5255585
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2835693, 35.2855682
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8505859, 32.8506546
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4269409, 34.4270630
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8054886, 43.7940903
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7993088, 37.7986984
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6605682, 36.6605530
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6661987, 37.6717606
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2838440, 57.2802429
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.4027939, 42.3908157
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9874039, 43.9858322
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6156998, 45.6103592
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2327347, 55.2328262
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2291183, 53.2253647
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5849457, 45.5852661
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8875351, 36.8872833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1428

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7853859, upper bound: 14.8127343
time: 37.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7834062, upper bound: 14.8147128
time: 31.42 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 71.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.29
Output dim: 2, lower bound: -14.8138652, upper bound: 14.7278167
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 71.29
Output dim: 2, lower bound: -14.7897050, upper bound: 14.7519687
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.29
Output dim: 2, lower bound: -14.8156554, upper bound: 14.7368414
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 71.29
Output dim: 2, lower bound: -14.8011035, upper bound: 14.7513966
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 71.29
Output dim: 2, lower bound: -14.7850964, upper bound: 14.8082956
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.29
Output dim: 2, lower bound: -14.7779724, upper bound: 14.8149862
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 71.29
Output dim: 2, lower bound: -14.7853859, upper bound: 14.8127343
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 71.29
Output dim: 2, lower bound: -14.7834062, upper bound: 14.8147128

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4320679, 54.4249039
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5621033, 36.5543175
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2360306, 33.2306023
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5258484, 32.5203171
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4925690, 43.4894638
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2601013, 38.2566833
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.4987030, 44.5056534
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7996292, 41.7882385
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1925049, 47.1866379
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8568192, 35.8524475
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6753693, 50.6763306
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5360565, 49.5262070
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2089310, 44.2252045
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6059113, 49.6097565
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5813446, 48.5673828
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6274796, 40.6292801
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9901657, 29.9823952
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1309814, 29.1335144
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4910355, 40.4799805
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5401840, 38.5497284
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2939682, 35.2889633
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8607407, 32.8587646
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4262695, 34.4263306
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8151093, 43.8268967
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7992783, 37.8004761
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6633224, 36.6625748
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6799469, 37.6712570
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2800446, 57.2834854
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3731461, 42.3873291
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -44.0158920, 44.0188065
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6003723, 45.6120148
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2652283, 55.2745514
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.1896210, 53.1995316
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5863724, 45.5887680
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8665085, 36.8698044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1649

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8093598, upper bound: 14.7207219
time: 35.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8067482, upper bound: 14.7233424
time: 33.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4311829, 54.4245911
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5732117, 36.5657501
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2340012, 33.2278976
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5180817, 32.5125961
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4946594, 43.4922256
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2536850, 38.2488251
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5335236, 44.5330124
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8020477, 41.7915573
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1922455, 47.1852112
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8513718, 35.8533249
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6215973, 50.6316605
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5099640, 49.5048370
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1862030, 44.2003021
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6138458, 49.6186523
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5723724, 48.5657425
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6302948, 40.6307983
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9599609, 29.9582138
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1391029, 29.1422081
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4426422, 40.4397888
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5425110, 38.5507965
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2837143, 35.2828751
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8603973, 32.8592796
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4214478, 34.4224091
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8150101, 43.8244476
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8198318, 37.8179283
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6662979, 36.6665573
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6730614, 37.6692810
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2723770, 57.2744217
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3912048, 42.4019623
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9884186, 43.9916534
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6068192, 45.6151047
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2689514, 55.2771378
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2248230, 53.2285385
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5953827, 45.5960388
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8921509, 36.8910217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 868

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 517

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8155359, upper bound: 14.7349455
time: 37.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8137555, upper bound: 14.7367197
time: 32.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3931274, 54.4030075
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5382690, 36.5458565
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2287674, 33.2329407
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5302124, 32.5321884
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4947281, 43.4925385
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2704391, 38.2735214
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5303955, 44.5305099
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8048553, 41.8110580
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1693039, 47.1738434
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8957138, 35.8908997
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6841736, 50.6787491
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5286026, 49.5359497
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2195206, 44.2109375
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5910416, 49.5932312
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5878296, 48.5990677
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6092186, 40.6044617
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9823380, 29.9882126
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1030159, 29.0953217
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4632568, 40.4662170
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5183868, 38.5101318
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2839279, 35.2863045
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8497047, 32.8492432
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4261627, 34.4268570
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7810669, 43.7632751
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7920761, 37.7886543
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6601868, 36.6604767
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6698227, 37.6767120
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2774048, 57.2760010
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.4034348, 42.3931656
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9724655, 43.9740372
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6223755, 45.6156540
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2207336, 55.2241364
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2307358, 53.2288513
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5831146, 45.5848541
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8888397, 36.8896103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1632

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1691

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7760772, upper bound: 14.8092138
time: 47.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7722010, upper bound: 14.8130423
time: 41.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4058838, 54.4136810
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5487747, 36.5548630
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2317123, 33.2359009
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5324173, 32.5343208
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4960861, 43.4944687
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2660294, 38.2691345
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5229034, 44.5241013
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7946014, 41.7991791
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1797256, 47.1836853
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8751907, 35.8696747
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6792068, 50.6711731
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5165710, 49.5202026
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2024612, 44.1951599
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5876617, 49.5899429
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5808182, 48.5860596
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6109962, 40.6087799
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9742508, 29.9771042
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1212883, 29.1154175
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4451675, 40.4451981
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5352325, 38.5297165
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2790833, 35.2795677
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8519402, 32.8511734
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4267654, 34.4269409
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8033752, 43.7918320
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7963791, 37.7954407
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6599197, 36.6597519
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6695938, 37.6731987
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2751007, 57.2740326
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3802719, 42.3709641
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9755936, 43.9762878
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6048126, 45.6009064
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2207718, 55.2227325
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2161636, 53.2142639
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5778732, 45.5794907
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8799057, 36.8809814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 948

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1370

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7851518, upper bound: 14.8125520
time: 37.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7852123, upper bound: 14.8124914
time: 35.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4054565, 54.4141159
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5482254, 36.5554199
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2311707, 33.2364426
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5320969, 32.5346413
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4953079, 43.4952545
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2657852, 38.2693787
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5247803, 44.5222244
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7934113, 41.8003845
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1794205, 47.1839981
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8736496, 35.8712158
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6778641, 50.6725235
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5151062, 49.5216675
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2041092, 44.1935272
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5882874, 49.5893250
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5789413, 48.5879440
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6111870, 40.6085892
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9726639, 29.9786949
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1202660, 29.1164436
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4427261, 40.4476471
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5356140, 38.5293427
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2775650, 35.2810822
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8511009, 32.8520126
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4268112, 34.4268913
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8032227, 43.7919846
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7960587, 37.7957573
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6597672, 36.6599045
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6676483, 37.6751404
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2776642, 57.2714691
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3829269, 42.3683090
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9778519, 43.9740219
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6062469, 45.5994568
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2226486, 55.2208633
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2180099, 53.2124176
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5791702, 45.5782013
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8812180, 36.8796692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 862

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 775

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7831149, upper bound: 14.8103881
time: 38.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7790810, upper bound: 14.8144208
time: 19.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 59.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.8093598, upper bound: 14.7207219
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.8067482, upper bound: 14.7233424
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.8155359, upper bound: 14.7349455
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.8137555, upper bound: 14.7367197
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.7760772, upper bound: 14.8092138
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.7722010, upper bound: 14.8130423
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.7851518, upper bound: 14.8125520
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.7852123, upper bound: 14.8124914
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.7831149, upper bound: 14.8103881
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 59.99
Output dim: 2, lower bound: -14.7790810, upper bound: 14.8144208

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4316711, 54.4243240
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5614243, 36.5533867
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2338486, 33.2266121
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5191269, 32.5075531
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4899292, 43.4849319
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2563858, 38.2495422
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.4976807, 44.5055771
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7987137, 41.7872086
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1911926, 47.1844482
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8562164, 35.8517456
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6731415, 50.6770782
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5316772, 49.5251694
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2061539, 44.2267838
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6072922, 49.6091080
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5776062, 48.5700302
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6255875, 40.6338120
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9824524, 29.9780922
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1260910, 29.1305923
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4856339, 40.4771576
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5402832, 38.5478592
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2873535, 35.2855263
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8570633, 32.8568115
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4185410, 34.4222336
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8137512, 43.8264160
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7977448, 37.8015518
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6586914, 36.6599731
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6681213, 37.6649704
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2832336, 57.2824097
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3723450, 42.3839035
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -44.0188217, 44.0175629
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6002884, 45.6118851
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2585754, 55.2712097
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.1876984, 53.1997986
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5863113, 45.5889969
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8649750, 36.8691788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 942

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 797

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8065403, upper bound: 14.7199868
time: 48.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8086205, upper bound: 14.7179024
time: 15.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4287415, 54.4226074
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5704422, 36.5636826
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2337646, 33.2277756
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5180359, 32.5124817
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4945984, 43.4921570
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2536850, 38.2487869
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5312653, 44.5315018
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8019714, 41.7918625
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1903992, 47.1840744
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8483276, 35.8490219
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6185532, 50.6269150
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5098495, 49.5047226
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1861725, 44.2002945
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6130066, 49.6173553
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5722046, 48.5657349
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6301041, 40.6307526
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9590607, 29.9569588
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1368790, 29.1395302
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4413147, 40.4376450
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5406799, 38.5482712
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2836990, 35.2827988
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8601608, 32.8590775
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4211197, 34.4221153
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8123322, 43.8203354
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8197861, 37.8180161
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6662827, 36.6665573
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6730537, 37.6692963
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2710648, 57.2734985
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3896866, 42.4008713
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9874115, 43.9907990
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6069641, 45.6150589
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2676239, 55.2760239
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2226410, 53.2270889
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5948486, 45.5956039
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8906937, 36.8899536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 965

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1664

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8149017, upper bound: 14.7342916
time: 38.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8149017, upper bound: 14.7342916
time: 42.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4291992, 54.4221573
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5711441, 36.5629807
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2338791, 33.2276611
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5179672, 32.5125504
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4945984, 43.4921722
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2536545, 38.2488174
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5319977, 44.5307541
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8023376, 41.7914810
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1911163, 47.1833649
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8470688, 35.8502769
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6168442, 50.6286240
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5098495, 49.5047150
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1861877, 44.2002869
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6125488, 49.6178131
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5723572, 48.5655823
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6302567, 40.6306038
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9587097, 29.9573174
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1364212, 29.1399841
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4404907, 40.4384689
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5399780, 38.5489655
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2836380, 35.2828598
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8601990, 32.8590431
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4211502, 34.4220772
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8108978, 43.8217697
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8199234, 37.8178787
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6663055, 36.6665421
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6730690, 37.6692810
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2714615, 57.2730865
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3901291, 42.4004288
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9875793, 43.9906464
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6067810, 45.6152573
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2678375, 55.2758179
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2233734, 53.2263641
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5949402, 45.5955124
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8910904, 36.8895645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1384

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8033624, upper bound: 14.7262918
time: 78.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8033624, upper bound: 14.7262918
time: 46.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3928528, 54.4019547
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5377121, 36.5464630
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2285614, 33.2329979
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5252991, 32.5332985
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4922562, 43.4914322
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2697525, 38.2743912
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5279999, 44.5329666
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8047028, 41.8104935
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1668015, 47.1730499
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8975906, 35.8890533
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6776581, 50.6685104
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5262680, 49.5281143
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2186813, 44.2100754
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5896149, 49.5915604
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5839081, 48.5888443
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6091881, 40.6043930
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9811478, 29.9787788
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1025620, 29.0931511
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4616928, 40.4562531
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5170288, 38.5095520
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2833252, 35.2801514
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8496170, 32.8489227
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4260559, 34.4248428
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7799835, 43.7613068
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7915039, 37.7872276
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6599808, 36.6592026
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6685104, 37.6690674
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2699890, 57.2747574
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3902893, 42.3907700
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9698181, 43.9730911
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6223450, 45.6156082
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2215729, 55.2237091
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2256699, 53.2280121
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5830383, 45.5848083
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8870010, 36.8901520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1559

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7634545, upper bound: 14.8087579
time: 41.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7756276, upper bound: 14.7965910
time: 42.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3920746, 54.4027328
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5388794, 36.5452957
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2288208, 33.2327385
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5313263, 32.5272713
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4936295, 43.4900589
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2713242, 38.2728271
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5328522, 44.5281143
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8043060, 41.8108978
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1685104, 47.1713409
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8938675, 35.8927727
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6739349, 50.6722488
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5207596, 49.5336075
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2186508, 44.2101135
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5893555, 49.5918121
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5776215, 48.5951385
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6091576, 40.6044197
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9729004, 29.9870300
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1008530, 29.0948563
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4533005, 40.4646454
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5178070, 38.5087814
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2777634, 35.2857056
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8493881, 32.8491440
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4241333, 34.4267693
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7790985, 43.7622070
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7906494, 37.7880859
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6589355, 36.6602478
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6621780, 37.6753998
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2761536, 57.2685776
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.4010315, 42.3800354
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9715118, 43.9714050
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6223145, 45.6156311
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2202911, 55.2249603
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2298813, 53.2237778
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5830688, 45.5847778
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8893814, 36.8877716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 929

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 965

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7715664, upper bound: 14.8125196
time: 34.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7716862, upper bound: 14.8123986
time: 35.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4048157, 54.4121857
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5477600, 36.5532608
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2311096, 33.2347336
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5321655, 32.5340385
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4953384, 43.4932938
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2658463, 38.2689362
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5194244, 44.5213089
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7945404, 41.7991257
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1783142, 47.1817932
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8711853, 35.8661232
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6711197, 50.6651688
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5148315, 49.5187988
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2021561, 44.1948318
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5872803, 49.5895538
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5775375, 48.5836487
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6053543, 40.6023445
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9730377, 29.9762268
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1206398, 29.1151428
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4440994, 40.4443970
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5344086, 38.5288925
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2787399, 35.2793427
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8513794, 32.8505783
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4265442, 34.4268723
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7994537, 43.7873688
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7944107, 37.7934494
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6596603, 36.6596680
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6684799, 37.6724319
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2726288, 57.2718658
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3794327, 42.3701782
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9754944, 43.9761429
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6043701, 45.6002197
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2199783, 55.2218933
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2137222, 53.2125854
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5752640, 45.5775681
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8765030, 36.8783875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7811820, upper bound: 14.8004239
time: 32.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7730164, upper bound: 14.8086022
time: 36.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4043579, 54.4126205
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5471954, 36.5538254
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2305527, 33.2353210
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5321350, 32.5340996
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4949112, 43.4937439
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2658310, 38.2689590
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5201111, 44.5206223
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7945557, 41.7991257
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1778259, 47.1822433
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8716431, 35.8656654
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6732101, 50.6630783
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5151672, 49.5184479
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2021408, 44.1948547
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5872803, 49.5895615
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5784225, 48.5827637
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6045609, 40.6033249
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9733658, 29.9758987
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1210136, 29.1147690
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4443741, 40.4441299
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5344086, 38.5288849
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2788620, 35.2792244
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8513412, 32.8506165
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4266968, 34.4267197
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7989197, 43.7879028
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7943726, 37.7934723
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6598358, 36.6594925
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6688309, 37.6720886
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2729187, 57.2715759
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3794937, 42.3701172
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9754486, 43.9761963
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6041260, 45.6004639
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2199936, 55.2219391
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2145157, 53.2118301
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5759506, 45.5768738
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8773117, 36.8775787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1445

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 780

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7840500, upper bound: 14.8114869
time: 45.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7842077, upper bound: 14.8113295
time: 37.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4054108, 54.4141006
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5484924, 36.5556793
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2307663, 33.2359848
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5328140, 32.5352707
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4934845, 43.4931030
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2665253, 38.2700729
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5148926, 44.5139542
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7945709, 41.8017883
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1793213, 47.1839676
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8722763, 35.8695831
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6782913, 50.6729202
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5155487, 49.5220947
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2049484, 44.1944351
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5866547, 49.5873795
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5792999, 48.5884552
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6068459, 40.6037827
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9718781, 29.9778709
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1193008, 29.1155014
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4399567, 40.4443893
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5328674, 38.5262222
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2787476, 35.2819901
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8509674, 32.8517685
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4267807, 34.4268646
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7996979, 43.7868271
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7962036, 37.7959328
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6597977, 36.6599350
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6677094, 37.6752090
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2718582, 57.2666397
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3788910, 42.3649368
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9779968, 43.9741440
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6059952, 45.5990372
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2214203, 55.2197342
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2102737, 53.2059402
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5747604, 45.5745163
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8735657, 36.8732758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 984

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 512

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7798880, upper bound: 14.8100280
time: 41.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7827480, upper bound: 14.8071731
time: 36.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4054108, 54.4141006
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5485001, 36.5556641
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2307205, 33.2360306
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5327225, 32.5353622
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4931641, 43.4934311
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2664948, 38.2700958
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5165100, 44.5123444
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7948151, 41.8015594
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1793823, 47.1839066
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8720169, 35.8698425
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6782303, 50.6729660
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5155029, 49.5221405
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2050247, 44.1943512
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5863495, 49.5876999
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5794525, 48.5883102
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6063728, 40.6042557
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9718323, 29.9779243
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1193161, 29.1154900
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4394684, 40.4448929
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5324860, 38.5266037
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2784805, 35.2822571
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8508606, 32.8518753
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4267807, 34.4268646
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7980804, 43.7884369
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7962341, 37.7959061
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6598053, 36.6599350
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6677094, 37.6752129
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2728043, 57.2656937
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3795471, 42.3642731
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9779816, 43.9741669
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6058426, 45.5991898
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2215118, 55.2196274
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2115402, 53.2046814
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5754623, 45.5737991
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8748169, 36.8720245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 516

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7790534, upper bound: 14.8143636
time: 36.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7790416, upper bound: 14.8143937
time: 42.46 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 81.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.8065403, upper bound: 14.7199868
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.8086205, upper bound: 14.7179024
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.8149017, upper bound: 14.7342916
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.8149017, upper bound: 14.7342916
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.8033624, upper bound: 14.7262918
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.8033624, upper bound: 14.7262918
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7634545, upper bound: 14.8087579
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7756276, upper bound: 14.7965910
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7715664, upper bound: 14.8125196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7716862, upper bound: 14.8123986
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7811820, upper bound: 14.8004239
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7730164, upper bound: 14.8086022
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7840500, upper bound: 14.8114869
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7842077, upper bound: 14.8113295
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7798880, upper bound: 14.8100280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7827480, upper bound: 14.8071731
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7790534, upper bound: 14.8143636
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 81.55
Output dim: 2, lower bound: -14.7790416, upper bound: 14.8143937

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4285431, 54.4220505
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5698090, 36.5624924
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2335663, 33.2270203
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5179138, 32.5111313
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4944305, 43.4918365
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2536850, 38.2478485
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5299072, 44.5300674
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8020172, 41.7913055
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1903839, 47.1833496
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8461761, 35.8476295
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6146774, 50.6248016
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5095596, 49.5048904
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1859970, 44.2012177
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6128998, 49.6172104
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5707703, 48.5650177
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6289062, 40.6308098
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9572296, 29.9559402
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1348114, 29.1382027
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4398651, 40.4371796
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5401154, 38.5479584
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2818680, 35.2818108
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8594284, 32.8587265
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4194183, 34.4210281
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8095093, 43.8190231
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8194733, 37.8178215
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6652985, 36.6659317
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6712685, 37.6684380
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2714081, 57.2727661
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3896484, 42.4005127
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9867935, 43.9896164
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6068726, 45.6149750
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2673645, 55.2759171
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2225647, 53.2270050
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5943298, 45.5950546
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8903503, 36.8895683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 994

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7967663, upper bound: 14.7160990
time: 33.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7967663, upper bound: 14.7160990
time: 37.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4281769, 54.4224243
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5692520, 36.5630493
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2330093, 33.2275772
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5166779, 32.5123672
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4942932, 43.4919815
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2527390, 38.2487793
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5298462, 44.5301437
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8014069, 41.7919159
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1896667, 47.1840591
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8469391, 35.8468742
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6164474, 50.6230392
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5100174, 49.5044479
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1870956, 44.2001190
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6128540, 49.6172485
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5715027, 48.5642929
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6301651, 40.6295547
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9580383, 29.9551277
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1355515, 29.1374588
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4408569, 40.4361801
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5403748, 38.5476990
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2827072, 35.2809753
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8598099, 32.8583527
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4200287, 34.4204178
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8110046, 43.8175125
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8195877, 37.8177032
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6656570, 36.6655731
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6721992, 37.6675034
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2703247, 57.2738495
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3893280, 42.4008408
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9862289, 43.9901810
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6069031, 45.6149445
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2675171, 55.2757721
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2225494, 53.2270050
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5942993, 45.5950851
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8903046, 36.8896179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1541

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8137419, upper bound: 14.7337424
time: 14.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8137419, upper bound: 14.7337355
time: 39.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3899078, 54.4005966
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5314102, 36.5387611
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2288208, 33.2336998
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5196457, 32.5174713
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4967651, 43.4944687
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2675629, 38.2697449
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5309448, 44.5261993
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8031845, 41.8098450
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1692352, 47.1729965
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8899384, 35.8875084
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6598358, 50.6542969
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5295258, 49.5406647
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2170334, 44.2082520
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5879745, 49.5899658
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5855255, 48.6001968
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6024170, 40.5984650
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9795380, 29.9914932
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1038284, 29.0975800
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4601212, 40.4698105
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5143585, 38.5054398
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2826004, 35.2893829
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8492928, 32.8490639
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4271774, 34.4290161
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7767029, 43.7597961
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7881927, 37.7859573
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6608963, 36.6617355
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6599846, 37.6717110
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2790833, 57.2719498
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3905869, 42.3706512
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9685516, 43.9681778
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6214676, 45.6147079
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2138367, 55.2173538
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2281799, 53.2219620
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5784683, 45.5793533
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8863678, 36.8843613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1664

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7709156, upper bound: 14.8118988
time: 43.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7709156, upper bound: 14.8118988
time: 43.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3899384, 54.4005508
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5323410, 36.5378227
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2297974, 33.2327309
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5215073, 32.5156021
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4980469, 43.4932022
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2682343, 38.2690811
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5309448, 44.5262070
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8032455, 41.8097687
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1701660, 47.1720734
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8885956, 35.8888474
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6559753, 50.6581497
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5278168, 49.5423889
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2168045, 44.2084961
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5875168, 49.5904312
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5826721, 48.6030350
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6032028, 40.5976753
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9773636, 29.9936638
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1035843, 29.0978279
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4584732, 40.4714584
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5144501, 38.5053329
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2814407, 35.2905502
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8493004, 32.8490524
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4263916, 34.4297981
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7766876, 43.7598190
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7885284, 37.7856255
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6604004, 36.6622314
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6584892, 37.6732063
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2795410, 57.2715073
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3916397, 42.3695908
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9683075, 43.9684219
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6213913, 45.6147842
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2127228, 55.2184753
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2280579, 53.2220840
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5776443, 45.5801773
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8859711, 36.8847618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1436

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7708197, upper bound: 14.8062825
time: 34.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7661397, upper bound: 14.8115350
time: 37.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4042740, 54.4125595
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5470047, 36.5536118
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2305374, 33.2353745
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5321350, 32.5342140
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4949112, 43.4938202
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2658310, 38.2691650
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5206909, 44.5204849
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7943115, 41.7990112
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1776199, 47.1820908
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8714142, 35.8662872
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6730423, 50.6629868
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5151672, 49.5184555
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2023697, 44.1947174
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5870972, 49.5898285
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5784760, 48.5825882
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6046562, 40.6030579
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9733200, 29.9759979
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1210136, 29.1149063
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4443741, 40.4445648
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5343933, 38.5288467
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2788086, 35.2792206
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8513565, 32.8505554
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4268036, 34.4265823
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7987671, 43.7880478
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7943192, 37.7933464
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6599121, 36.6594162
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6688957, 37.6719856
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2730942, 57.2715683
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3802643, 42.3701019
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9754486, 43.9761963
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6039886, 45.6002808
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2203064, 55.2219315
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2153320, 53.2117996
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5760269, 45.5768738
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8777771, 36.8775673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1584

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7836911, upper bound: 14.8113951
time: 31.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7839714, upper bound: 14.8111063
time: 19.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4043045, 54.4126205
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5469894, 36.5538254
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2305527, 33.2352982
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5321350, 32.5340996
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4949112, 43.4937515
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2658310, 38.2689514
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5199738, 44.5206223
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7944489, 41.7991257
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1776962, 47.1822433
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8716431, 35.8654480
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6732101, 50.6629181
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5151672, 49.5184479
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2020035, 44.1948547
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5872803, 49.5893555
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5782166, 48.5827637
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6042976, 40.6033249
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9733658, 29.9758492
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1210136, 29.1147690
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4443741, 40.4441223
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5344086, 38.5288773
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2788620, 35.2791786
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8512802, 32.8506165
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4265518, 34.4267197
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7989197, 43.7877579
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7942505, 37.7934723
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6597595, 36.6594925
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6687202, 37.6720886
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2729111, 57.2715759
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3794861, 42.3701172
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9754486, 43.9761963
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6041260, 45.6003418
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2199707, 55.2219391
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2144775, 53.2118301
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5759659, 45.5768738
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8773041, 36.8775787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1621

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7438556, upper bound: 14.8104245
time: 37.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7833029, upper bound: 14.7709673
time: 38.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4012146, 54.4103012
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5416489, 36.5495529
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2255859, 33.2313538
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5254593, 32.5287933
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4909210, 43.4910965
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2648926, 38.2688751
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5097656, 44.5082474
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7957916, 41.8035736
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1739578, 47.1791534
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8705368, 35.8674355
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6723785, 50.6660538
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5139236, 49.5202637
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1996231, 44.1881638
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5849609, 49.5854568
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5772095, 48.5860825
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6068878, 40.6040535
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9727936, 29.9784355
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1136894, 29.1094398
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4392700, 40.4436188
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5333099, 38.5266724
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2796936, 35.2827225
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8506241, 32.8514709
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4252167, 34.4250107
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7997665, 43.7875061
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7961960, 37.7961082
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6594543, 36.6594620
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6673126, 37.6745148
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2715607, 57.2660370
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3790741, 42.3649597
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9771576, 43.9729614
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6038589, 45.5964508
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2207489, 55.2182541
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2040176, 53.1991806
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5764465, 45.5758591
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8751755, 36.8744202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1462

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7792194, upper bound: 14.8064609
time: 39.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7755625, upper bound: 14.8092535
time: 31.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4044189, 54.4136963
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5474319, 36.5553169
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2304840, 33.2361526
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5326996, 32.5355721
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4931641, 43.4935455
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2664948, 38.2701035
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5164490, 44.5121994
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7947845, 41.8016586
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1786652, 47.1837845
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8717194, 35.8683395
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6776886, 50.6706772
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5153885, 49.5218201
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2049866, 44.1943588
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5862122, 49.5875473
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5791779, 48.5879364
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6063347, 40.6047668
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9716721, 29.9773407
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1189766, 29.1143532
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4392929, 40.4442139
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5321350, 38.5259781
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2785263, 35.2822342
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8507690, 32.8518219
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4267502, 34.4268188
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7976913, 43.7877579
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7962265, 37.7959251
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6597519, 36.6598663
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6675377, 37.6749496
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2727737, 57.2656250
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3794327, 42.3641357
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9773331, 43.9739151
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6057892, 45.5991898
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2211304, 55.2194748
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2115479, 53.2046738
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5756226, 45.5737839
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8749695, 36.8719978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 895

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 796

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7756629, upper bound: 14.8143146
time: 46.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7790045, upper bound: 14.8109714
time: 40.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4049988, 54.4131088
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5481567, 36.5545998
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2308426, 33.2357941
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5329437, 32.5353355
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4932709, 43.4934311
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2665100, 38.2700806
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5163574, 44.5122757
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7949066, 41.8015366
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1792603, 47.1831970
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8705139, 35.8695488
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6759491, 50.6723938
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5152054, 49.5220108
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2050171, 44.1943359
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5861969, 49.5875549
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5790863, 48.5880203
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6068764, 40.6042213
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9712448, 29.9777718
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1181831, 29.1151390
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4388046, 40.4447250
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5318756, 38.5262527
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2784500, 35.2823105
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8508148, 32.8517838
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4267349, 34.4268303
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7973862, 43.7880478
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7962570, 37.7958984
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6597290, 36.6598969
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6674385, 37.6750488
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2727585, 57.2656326
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3794174, 42.3641434
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9777298, 43.9735184
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6058502, 45.5991516
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2213593, 55.2192535
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2115326, 53.2046814
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5754547, 45.5739517
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8748169, 36.8721504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1412

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1445

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7789448, upper bound: 14.8114031
time: 38.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7760495, upper bound: 14.8142965
time: 37.68 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 78.17 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7967663, upper bound: 14.7160990
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7967663, upper bound: 14.7160990
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.8137419, upper bound: 14.7337424
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.8137419, upper bound: 14.7337355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7709156, upper bound: 14.8118988
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7709156, upper bound: 14.8118988
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7708197, upper bound: 14.8062825
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7661397, upper bound: 14.8115350
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7836911, upper bound: 14.8113951
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7839714, upper bound: 14.8111063
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7438556, upper bound: 14.8104245
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7833029, upper bound: 14.7709673
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7792194, upper bound: 14.8064609
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7755625, upper bound: 14.8092535
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7756629, upper bound: 14.8143146
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7790045, upper bound: 14.8109714
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7789448, upper bound: 14.8114031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 78.17
Output dim: 2, lower bound: -14.7760495, upper bound: 14.8142965

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4265747, 54.4204712
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5685577, 36.5617142
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2342911, 33.2285156
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5192261, 32.5146523
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4943542, 43.4920425
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2549286, 38.2507782
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5226898, 44.5219040
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8053589, 41.7951431
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1905212, 47.1841660
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8415833, 35.8421860
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6118774, 50.6199036
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5096283, 49.5040741
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1900330, 44.2033539
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6119385, 49.6169968
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5715790, 48.5643768
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6276779, 40.6270981
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9547844, 29.9523697
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1336250, 29.1357384
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4361115, 40.4322128
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5360413, 38.5440598
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2842865, 35.2827759
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8597870, 32.8583221
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4197845, 34.4201927
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8028946, 43.8106995
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8223114, 37.8199310
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6653671, 36.6652985
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6713181, 37.6666527
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2676697, 57.2709045
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3885880, 42.4001083
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9876862, 43.9917526
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6083221, 45.6166992
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2689438, 55.2773056
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2177124, 53.2215042
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5921402, 45.5928421
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8847046, 36.8833771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 878

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8118157, upper bound: 14.7260292
time: 38.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8060233, upper bound: 14.7318262
time: 35.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4262390, 54.4209137
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5679169, 36.5623932
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2339478, 33.2288818
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5189667, 32.5149727
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4943542, 43.4920502
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2547455, 38.2509689
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5215912, 44.5231552
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8046265, 41.7960129
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1897888, 47.1848602
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8423538, 35.8415222
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6132355, 50.6184692
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5096436, 49.5040665
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1903381, 44.2030487
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6126556, 49.6163406
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5715790, 48.5643768
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6277008, 40.6270714
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9553185, 29.9518738
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1337929, 29.1355324
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4369049, 40.4314346
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5367584, 38.5433731
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2846680, 35.2825546
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8597794, 32.8583298
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4197998, 34.4201736
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8042984, 43.8093796
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8218231, 37.8204422
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6653748, 36.6652908
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6713409, 37.6666298
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2673798, 57.2712173
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3885880, 42.4001160
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9877930, 43.9916382
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6086578, 45.6163864
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2690506, 55.2771912
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2170410, 53.2222214
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5920639, 45.5930099
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8840637, 36.8841209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1606

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1462

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8140563, upper bound: 14.7294857
time: 36.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8102845, upper bound: 14.7332600
time: 36.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3896942, 54.4000549
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5307693, 36.5375710
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2286224, 33.2329521
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5195389, 32.5161209
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4965973, 43.4941483
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2675629, 38.2688065
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5296021, 44.5247726
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8032227, 41.8092880
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1692276, 47.1722794
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8877945, 35.8861198
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6559601, 50.6521988
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5292664, 49.5408401
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2168732, 44.2091599
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5878906, 49.5898209
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5840912, 48.5994873
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6012192, 40.5985184
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9777069, 29.9904709
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1017609, 29.0962563
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4586563, 40.4693604
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5137939, 38.5051498
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2807693, 35.2883759
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8485680, 32.8487167
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4254761, 34.4279175
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7738800, 43.7584915
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7878647, 37.7857513
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6599197, 36.6611099
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6581993, 37.6708603
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2794342, 57.2712173
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3905487, 42.3702850
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9679184, 43.9669724
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6213455, 45.6146164
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2135773, 55.2172623
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2281036, 53.2218628
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5779419, 45.5787964
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8860168, 36.8839760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 880

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1479

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7685978, upper bound: 14.8048823
time: 37.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7638921, upper bound: 14.8096003
time: 78.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3893280, 54.4004211
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5302200, 36.5381279
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2280731, 33.2335091
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5183029, 32.5173569
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4964447, 43.4942932
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2666321, 38.2697296
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5295258, 44.5248489
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.8026123, 41.8099060
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1685104, 47.1729889
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8885498, 35.8853645
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6577301, 50.6504364
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5297089, 49.5404053
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2179718, 44.2080688
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5878296, 49.5898666
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5848083, 48.5987625
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6024704, 40.5972672
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9785156, 29.9896622
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1025085, 29.0955124
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4596634, 40.4683533
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5140533, 38.5048828
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2816010, 35.2875404
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8489418, 32.8483429
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4260788, 34.4273071
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7753906, 43.7569809
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7879791, 37.7856293
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6602707, 36.6607590
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6591301, 37.6699257
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2783508, 57.2723007
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3902283, 42.3706207
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9673538, 43.9675369
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6213760, 45.6145859
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2137299, 55.2171097
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2280884, 53.2218704
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5779114, 45.5788193
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8859711, 36.8840256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 774

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7708452, upper bound: 14.8082226
time: 50.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7672329, upper bound: 14.8118282
time: 37.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3882599, 54.3993149
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5311737, 36.5369682
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2280884, 33.2314415
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5207825, 32.5147858
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4977875, 43.4930573
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2664642, 38.2676086
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5290527, 44.5235977
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7995987, 41.8070374
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1695023, 47.1716156
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8843231, 35.8856506
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6529236, 50.6555023
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5230560, 49.5389099
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2123718, 44.2025452
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5870895, 49.5898972
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5764084, 48.5986710
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6021881, 40.5968132
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9730301, 29.9904099
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1024704, 29.0969048
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4525299, 40.4668732
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5126190, 38.5027237
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2786484, 35.2884064
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8486481, 32.8485718
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4263229, 34.4297447
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7742920, 43.7572937
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7882309, 37.7854462
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6600494, 36.6619492
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6544418, 37.6703568
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2765808, 57.2677002
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3845291, 42.3604813
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9653320, 43.9645309
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6183548, 45.6109543
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2092285, 55.2138672
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2246704, 53.2175598
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5771408, 45.5794373
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8854141, 36.8839912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1543

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 763

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7625155, upper bound: 14.8090526
time: 34.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7636463, upper bound: 14.8080399
time: 36.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4042587, 54.4121780
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5497971, 36.5560455
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2320709, 33.2365799
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5319366, 32.5340805
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4954300, 43.4943161
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2669067, 38.2700806
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5171509, 44.5165634
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7963867, 41.8010559
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1796570, 47.1836472
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8642807, 35.8603363
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6598892, 50.6520157
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5140305, 49.5174408
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2021255, 44.1944580
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5847473, 49.5875702
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5754089, 48.5797653
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6095200, 40.6075401
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9748383, 29.9780083
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1210938, 29.1150208
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4428329, 40.4435883
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5332260, 38.5278702
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2790756, 35.2795258
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8508987, 32.8501053
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4266663, 34.4264603
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7983246, 43.7875519
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7916641, 37.7898941
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6598892, 36.6594009
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6687698, 37.6718750
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2767639, 57.2755814
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3839645, 42.3745804
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9798050, 43.9808731
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6028976, 45.5991287
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2215652, 55.2231827
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2139359, 53.2102432
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5767212, 45.5776062
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8759460, 36.8755722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 795

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 771

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7824543, upper bound: 14.8111529
time: 50.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7834681, upper bound: 14.8101631
time: 40.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4038925, 54.4125595
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5494385, 36.5564117
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2317352, 33.2369156
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5319977, 32.5340118
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4954147, 43.4943314
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2667236, 38.2702484
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5167694, 44.5169449
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7963562, 41.8010941
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1791687, 47.1841354
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8654709, 35.8591423
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6620712, 50.6498260
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5141525, 49.5173111
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2021255, 44.1944656
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5848389, 49.5875015
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5756683, 48.5794983
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6091385, 40.6079254
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9753265, 29.9775162
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1211243, 29.1149902
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4433823, 40.4430389
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5334091, 38.5276718
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2791290, 35.2794724
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8509064, 32.8500900
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4266739, 34.4264450
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7982635, 43.7875977
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7908707, 37.7906876
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6598892, 36.6593933
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6687775, 37.6718636
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2770996, 57.2752380
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3847427, 42.3738098
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9801254, 43.9805450
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6028366, 45.5991898
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2215347, 55.2231979
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2137527, 53.2104187
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5767365, 45.5775909
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8757782, 36.8757324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1448

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 971

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7728971, upper bound: 14.8094563
time: 42.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7831004, upper bound: 14.7978232
time: 33.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3692780, 54.3833313
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5405807, 36.5571442
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.1593552, 33.1746635
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4550247, 32.4757118
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4665527, 43.4717560
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.1890945, 38.2042923
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5300293, 44.5347595
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7704468, 41.7772827
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1079102, 47.1234970
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7867661, 35.7646484
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.4953308, 50.4510651
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4421463, 49.4317627
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1307220, 44.1159286
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5880508, 49.5903015
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4829178, 48.4642792
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.5958977, 40.5912476
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9690399, 29.9539299
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.0699005, 29.0510864
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.3358459, 40.3172455
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5333557, 38.5279617
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2717743, 35.2592163
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8656235, 32.8614120
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4373093, 34.4277458
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7431946, 43.7253342
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7937546, 37.7928696
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6738281, 36.6675568
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.7241058, 37.7089844
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2048798, 57.2147675
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3329849, 42.3324585
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9105530, 43.9221115
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6050415, 45.6022263
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2217865, 55.2211609
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2141266, 53.2114716
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5720291, 45.5729065
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8821640, 36.8823433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 980

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 784

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7405229, upper bound: 14.8103088
time: 41.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7437401, upper bound: 14.8071001
time: 41.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.3892975, 54.3998260
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5390320, 36.5477982
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2217026, 33.2284775
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5264587, 32.5299110
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4908066, 43.4910049
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2665939, 38.2710953
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5120697, 44.5099640
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7977142, 41.8058777
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1682434, 47.1742859
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8632278, 35.8588791
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6512909, 50.6423035
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5153122, 49.5217285
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1987991, 44.1869125
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5812759, 49.5817032
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5749207, 48.5836182
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6117058, 40.6101646
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9723473, 29.9778900
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1089745, 29.1040688
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4394684, 40.4437714
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5366821, 38.5293198
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2772903, 35.2802887
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8506012, 32.8514557
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4245071, 34.4241943
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7987137, 43.7863388
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7978821, 37.7983551
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6590424, 36.6588593
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6672707, 37.6744614
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2834473, 57.2761612
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3974152, 42.3802795
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9793091, 43.9750290
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5984421, 45.5914612
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2145233, 55.2122498
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2044601, 53.1985321
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5788574, 45.5779419
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8782806, 36.8768005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1776

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 980

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7663471, upper bound: 14.7892990
time: 35.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7555992, upper bound: 14.8000469
time: 39.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4041748, 54.4136429
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5469971, 36.5550575
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2303238, 33.2363663
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5326309, 32.5357590
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4930878, 43.4934998
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2664642, 38.2705536
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5172272, 44.5119019
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7947617, 41.8016510
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1783600, 47.1836014
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8706055, 35.8684502
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6771393, 50.6697617
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5153732, 49.5225525
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2057495, 44.1940536
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5849457, 49.5874786
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5790100, 48.5881653
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6060905, 40.6032066
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9713821, 29.9780235
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1188889, 29.1141396
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4392624, 40.4454117
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5321274, 38.5259247
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2782974, 35.2821274
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8507004, 32.8515015
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4266891, 34.4263573
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7976685, 43.7877197
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7959213, 37.7949867
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6596985, 36.6595993
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6675224, 37.6748581
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2728577, 57.2656250
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3813782, 42.3640823
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9768448, 43.9735107
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6057587, 45.5991364
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2210999, 55.2193604
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2131653, 53.2046280
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5755997, 45.5737839
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8752899, 36.8719788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7359077, upper bound: 14.8124323
time: 39.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7737785, upper bound: 14.7745739
time: 37.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4044189, 54.4134598
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5471725, 36.5553169
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2304840, 33.2360001
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5326996, 32.5354996
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4931183, 43.4935455
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2664948, 38.2700806
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5161591, 44.5121994
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7947617, 41.8016586
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1784973, 47.1837845
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8717194, 35.8672256
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6776886, 50.6701355
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5153885, 49.5218048
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2046967, 44.1943588
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5862122, 49.5862961
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5791779, 48.5877838
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6047707, 40.6047668
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9716721, 29.9770470
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1187592, 29.1143532
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4392929, 40.4441757
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5320816, 38.5259781
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2785263, 35.2820053
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8504410, 32.8518219
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4263000, 34.4268188
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7976379, 43.7877579
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7952881, 37.7959251
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6594849, 36.6598663
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6675377, 37.6749344
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2727661, 57.2656250
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3793793, 42.3641357
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9773331, 43.9734268
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6057892, 45.5991592
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2211304, 55.2194214
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2114868, 53.2046738
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5756302, 45.5737839
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8749390, 36.8719978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1780

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1568

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7779011, upper bound: 14.8098187
time: 44.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7778594, upper bound: 14.8098645
time: 42.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4047241, 54.4128571
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5478973, 36.5541878
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2307739, 33.2356415
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5327301, 32.5350990
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4932404, 43.4933624
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2662811, 38.2698059
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5162354, 44.5122375
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7939758, 41.8000031
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1788635, 47.1826935
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8686829, 35.8661995
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6732178, 50.6677170
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5130310, 49.5192108
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.2022247, 44.1926651
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5853882, 49.5870438
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5761490, 48.5842209
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6056633, 40.6034927
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9689102, 29.9751282
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1164627, 29.1130829
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4364319, 40.4411697
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5314178, 38.5260239
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2770157, 35.2807541
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8506165, 32.8515320
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4262695, 34.4264259
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7961807, 43.7872086
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7960434, 37.7957077
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6592178, 36.6594467
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6646805, 37.6718636
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2719955, 57.2649384
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3764420, 42.3616104
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9746933, 43.9709167
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.6029205, 45.5973358
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2183380, 55.2173691
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2105865, 53.2040329
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5753479, 45.5738907
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8740921, 36.8715820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7779834, upper bound: 14.7903646
time: 38.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7579085, upper bound: 14.8104451
time: 33.84 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 74.53 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.8118157, upper bound: 14.7260292
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.8060233, upper bound: 14.7318262
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.8140563, upper bound: 14.7294857
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.8102845, upper bound: 14.7332600
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7685978, upper bound: 14.8048823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7638921, upper bound: 14.8096003
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7708452, upper bound: 14.8082226
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7672329, upper bound: 14.8118282
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7625155, upper bound: 14.8090526
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7636463, upper bound: 14.8080399
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7824543, upper bound: 14.8111529
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7834681, upper bound: 14.8101631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7728971, upper bound: 14.8094563
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7831004, upper bound: 14.7978232
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7405229, upper bound: 14.8103088
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7437401, upper bound: 14.8071001
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7663471, upper bound: 14.7892990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7555992, upper bound: 14.8000469
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7359077, upper bound: 14.8124323
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7737785, upper bound: 14.7745739
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7779011, upper bound: 14.8098187
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7778594, upper bound: 14.8098645
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7779834, upper bound: 14.7903646
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 74.53
Output dim: 2, lower bound: -14.7579085, upper bound: 14.8104451
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 74.53
Output dim: 2, lower bound: -14.7760495, upper bound: 14.8142965

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 49.25 + 3588.56 = 3637.81 seconds
