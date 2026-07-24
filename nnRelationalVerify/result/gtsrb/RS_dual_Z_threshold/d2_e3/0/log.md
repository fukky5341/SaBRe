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
execution time: IAR + RelationalAnalysis = 2.71 + 47.25 = 49.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -14.8236368, upper bound: 14.8236369

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7772129, upper bound: 14.8220449
time: 37.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8220448, upper bound: 14.7772129
time: 34.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 72.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 72.40
Output dim: 2, lower bound: -14.7772129, upper bound: 14.8220449
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 72.40
Output dim: 2, lower bound: -14.8220448, upper bound: 14.7772129

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4363403, 54.4369431
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5578537, 36.5598869
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2274323, 33.2302589
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5091400, 32.5132065
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4829483, 43.4856339
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2432251, 38.2468338
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5403595, 44.5407562
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7633057, 41.7700043
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1783600, 47.1809845
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8348846, 35.8399925
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6872559, 50.6874847
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5185852, 49.5219040
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1599655, 44.1495209
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6284103, 49.6238632
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5690689, 48.5722961
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6400909, 40.6374474
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9826889, 29.9839630
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1611671, 29.1649323
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4729767, 40.4770508
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5354080, 38.5326080
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2892532, 35.2903595
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8621330, 32.8634720
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4271088, 34.4270020
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8252945, 43.8226166
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8104706, 37.8147163
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6656036, 36.6656570
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6876869, 37.6907425
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2712860, 57.2680969
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3627396, 42.3546295
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9699020, 43.9616928
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5945663, 45.5888901
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2477264, 55.2362518
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2260971, 53.2231445
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5925064, 45.5915451
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8993301, 36.9012146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7702758, upper bound: 14.7916132
time: 33.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7467707, upper bound: 14.8151219
time: 52.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4369354, 54.4363480
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5598907, 36.5578575
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2302551, 33.2274361
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5132065, 32.5091400
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4856339, 43.4829483
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2468262, 38.2432327
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5407562, 44.5403671
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7700043, 41.7632980
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1809845, 47.1783676
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8399887, 35.8348846
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6874847, 50.6872559
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5219116, 49.5185852
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1495285, 44.1599579
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6238632, 49.6284027
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5723038, 48.5690536
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6374435, 40.6400909
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9839630, 29.9826927
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1649361, 29.1611633
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4770508, 40.4729843
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5326004, 38.5354080
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2903671, 35.2892494
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8634758, 32.8621292
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4270020, 34.4271126
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8226089, 43.8252945
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8147125, 37.8104706
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6656570, 36.6656036
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6907387, 37.6876831
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2680969, 57.2712860
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3546219, 42.3627396
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9616928, 43.9699020
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5888901, 45.5945663
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2362518, 55.2477264
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2231369, 53.2260895
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5915604, 45.5924988
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9012146, 36.8993301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8151218, upper bound: 14.7467708
time: 30.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7916131, upper bound: 14.7702759
time: 42.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 75.81 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 75.81
Output dim: 2, lower bound: -14.7702758, upper bound: 14.7916132
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 75.81
Output dim: 2, lower bound: -14.7467707, upper bound: 14.8151219
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 75.81
Output dim: 2, lower bound: -14.8151218, upper bound: 14.7467708
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 75.81
Output dim: 2, lower bound: -14.7916131, upper bound: 14.7702759

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4326019, 54.4369431
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5539703, 36.5598869
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2248993, 33.2302589
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5074692, 32.5132065
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4824524, 43.4856339
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2411041, 38.2468338
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5403595, 44.5404892
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7580109, 41.7700043
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1764069, 47.1809845
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8344498, 35.8399925
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6865234, 50.6874847
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5148697, 49.5219040
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1599655, 44.1413116
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6284103, 49.6222153
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5629120, 48.5722961
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6400909, 40.6367226
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9790955, 29.9839630
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1610641, 29.1649323
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4699097, 40.4770508
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5354080, 38.5298843
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2870026, 35.2903595
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8609695, 32.8634720
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4264374, 34.4270020
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8252945, 43.8146515
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8096924, 37.8147163
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6649628, 36.6656570
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6816788, 37.6907425
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2712860, 57.2652512
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3627396, 42.3457947
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9699020, 43.9569168
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5945663, 45.5822906
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2477264, 55.2306747
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2260971, 53.2204208
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5925064, 45.5913925
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8990097, 36.9012146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7009470, upper bound: 14.8133181
time: 31.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7449636, upper bound: 14.7692951
time: 44.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4377289, 54.4325943
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5607147, 36.5539742
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2308960, 33.2249031
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5137253, 32.5074692
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4860229, 43.4824524
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2469788, 38.2411041
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5404816, 44.5404053
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7732544, 41.7580109
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1830902, 47.1764069
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8424988, 35.8344536
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6861572, 50.6865234
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5259781, 49.5148621
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1413040, 44.1625900
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6222153, 49.6309280
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5778809, 48.5629120
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6367188, 40.6397285
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9859238, 29.9790955
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1648331, 29.1610565
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4811249, 40.4699097
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5298843, 38.5401917
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2917023, 35.2870064
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8638763, 32.8609657
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4270477, 34.4264374
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8146439, 43.8256607
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8152161, 37.8096924
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6660690, 36.6649628
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6923599, 37.6816788
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2652435, 57.2713928
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3457870, 42.3644104
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9569244, 43.9700241
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5822983, 45.5958939
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2306671, 55.2482452
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2204208, 53.2264862
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5914078, 45.5925064
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9017410, 36.8990097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7692951, upper bound: 14.7449637
time: 41.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8133180, upper bound: 14.7009470
time: 36.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 80.10 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 80.10
Output dim: 2, lower bound: -14.7009470, upper bound: 14.8133181
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 80.10
Output dim: 2, lower bound: -14.7449636, upper bound: 14.7692951
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 80.10
Output dim: 2, lower bound: -14.7692951, upper bound: 14.7449637
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 80.10
Output dim: 2, lower bound: -14.8133180, upper bound: 14.7009470

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4276276, 54.4340973
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5479279, 36.5581093
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2155762, 33.2269936
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4875488, 32.5070877
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4794006, 43.4846344
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2273254, 38.2426910
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5373993, 44.5425186
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7515411, 41.7653122
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1680298, 47.1784973
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8332443, 35.8337822
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6787567, 50.6644135
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5107727, 49.5147552
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1531296, 44.1231689
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6254807, 49.6207275
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5634384, 48.5656967
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6368332, 40.6170425
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9776917, 29.9750214
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1576691, 29.1521339
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4650192, 40.4632568
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5327377, 38.5266800
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2869720, 35.2825241
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8611908, 32.8595428
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4254303, 34.4188309
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8171387, 43.7924194
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8085022, 37.8090553
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6671219, 36.6637039
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6855927, 37.6844215
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2540741, 57.2595978
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3554306, 42.3417282
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9553299, 43.9515915
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5936279, 45.5813599
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2470474, 55.2284317
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2252579, 53.2198715
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5911331, 45.5921021
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8983841, 36.9027176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7006348, upper bound: 14.8133131
time: 41.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7009412, upper bound: 14.8130060
time: 36.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4348907, 54.4276352
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5589447, 36.5479317
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2276306, 33.2155724
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5076065, 32.4875488
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4850311, 43.4794006
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2428436, 38.2273254
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5425110, 44.5374298
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7685547, 41.7515411
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1806030, 47.1680298
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8362656, 35.8332481
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6630859, 50.6787567
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5188293, 49.5107651
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1231766, 44.1557693
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6207199, 49.6279984
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5712967, 48.5634460
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6170425, 40.6364746
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9769745, 29.9776917
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1520386, 29.1576653
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4673233, 40.4650269
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5266800, 38.5375137
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2838821, 35.2869759
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8599396, 32.8611908
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4188843, 34.4254303
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7924194, 43.8175049
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8095627, 37.8085022
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6641159, 36.6671219
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6860428, 37.6855927
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2595978, 57.2541962
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3417206, 42.3570938
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9515915, 43.9554443
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5813446, 45.5949554
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2284317, 55.2475739
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2198715, 53.2256775
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5921097, 45.5911331
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9032440, 36.8983841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8130059, upper bound: 14.7009412
time: 38.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8133130, upper bound: 14.7006349
time: 41.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 82.52 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 82.52
Output dim: 2, lower bound: -14.7006348, upper bound: 14.8133131
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 82.52
Output dim: 2, lower bound: -14.7009412, upper bound: 14.8130060
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 82.52
Output dim: 2, lower bound: -14.8130059, upper bound: 14.7009412
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 82.52
Output dim: 2, lower bound: -14.8133130, upper bound: 14.7006349

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4275818, 54.4340744
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5503311, 36.5613823
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2106400, 33.2235413
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4784851, 32.4996414
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4724731, 43.4787292
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2178345, 38.2362595
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5464783, 44.5565643
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7600861, 41.7768097
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1573868, 47.1698761
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8334274, 35.8339310
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6576691, 50.6405106
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5117188, 49.5155792
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1298599, 44.0947571
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5958710, 49.5896988
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5625534, 48.5646896
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6409645, 40.6204605
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9670410, 29.9630432
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1660080, 29.1609230
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4580154, 40.4554672
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5324249, 38.5264359
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2822495, 35.2770996
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8598480, 32.8580170
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4238358, 34.4169006
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8188477, 43.7935333
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7800369, 37.7811661
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6664124, 36.6629639
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6865273, 37.6852036
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2549210, 57.2604904
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3673477, 42.3533249
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9687271, 43.9638138
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5941162, 45.5814362
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2601242, 55.2353058
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2203369, 53.2166748
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5913544, 45.5925446
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9064255, 36.9142723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6996425, upper bound: 14.7922723
time: 43.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6795541, upper bound: 14.8123530
time: 31.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4275970, 54.4340668
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5512009, 36.5605125
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2121201, 33.2220612
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4801025, 32.4980240
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4734802, 43.4777069
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2208862, 38.2332001
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5514526, 44.5516129
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7630310, 41.7738647
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1594009, 47.1678696
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8334045, 35.8339539
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6548615, 50.6433334
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5115814, 49.5157318
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1247330, 44.0998917
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5944366, 49.5911484
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5624161, 48.5648117
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6402473, 40.6211700
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9657135, 29.9643669
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1664505, 29.1604729
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4572372, 40.4562454
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5324860, 38.5263901
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2815552, 35.2777939
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8596725, 32.8581924
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4235077, 34.4172287
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8182526, 43.7941360
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7806091, 37.7805939
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6663742, 36.6630096
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6863670, 37.6853600
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2549667, 57.2604523
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3670273, 42.3536453
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9675369, 43.9650040
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5936890, 45.5818481
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2539291, 55.2414856
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2220764, 53.2149353
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5915680, 45.5923309
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9099426, 36.9107590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6999494, upper bound: 14.7919653
time: 38.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6798601, upper bound: 14.8120470
time: 33.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4348450, 54.4275970
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5613480, 36.5512047
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2226944, 33.2121201
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4985352, 32.4801102
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4781036, 43.4734802
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2333374, 38.2208862
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5516052, 44.5514755
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7770996, 41.7630310
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1699600, 47.1593933
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8364487, 35.8334045
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6419983, 50.6548538
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5197906, 49.5115738
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.0998917, 44.1273575
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5911407, 49.5969696
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5704117, 48.5624237
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6211739, 40.6398926
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9663239, 29.9657135
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1603775, 29.1664543
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4603043, 40.4572449
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5263824, 38.5372696
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2791519, 35.2815552
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8585968, 32.8596725
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4172897, 34.4235077
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7941437, 43.8186188
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7810898, 37.7806053
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6634064, 36.6663742
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6869850, 37.6863708
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2604599, 57.2550812
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3536377, 42.3686905
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9649963, 43.9676590
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5818481, 45.5950317
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2414932, 55.2544403
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2149353, 53.2224808
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5923309, 45.5915756
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9112854, 36.9099426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7919652, upper bound: 14.6798602
time: 34.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7919653, upper bound: 14.6999495
time: 32.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4348602, 54.4275894
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5622177, 36.5503349
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2241745, 33.2106400
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5001678, 32.4784851
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4791260, 43.4724731
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2364044, 38.2178268
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5565796, 44.5465240
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7800598, 41.7600861
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1719742, 47.1573868
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8364258, 35.8334351
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6391754, 50.6576767
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5196381, 49.5117264
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.0947647, 44.1324921
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5897064, 49.5984192
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5702744, 48.5625458
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6204567, 40.6406021
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9649963, 29.9670410
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1608353, 29.1660042
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4595413, 40.4580154
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5264435, 38.5372238
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2784576, 35.2822495
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8584213, 32.8598442
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4169540, 34.4238396
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7935333, 43.8192215
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7816620, 37.7800369
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6633682, 36.6664124
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6868248, 37.6865273
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2604904, 57.2550430
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3533325, 42.3690109
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9638062, 43.9688492
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5814514, 45.5954437
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2352982, 55.2606277
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2166748, 53.2207413
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5925446, 45.5913620
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9148026, 36.9064255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8123529, upper bound: 14.6795541
time: 39.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7922722, upper bound: 14.6996426
time: 33.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 75.49 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 75.49
Output dim: 2, lower bound: -14.6996425, upper bound: 14.7922723
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 75.49
Output dim: 2, lower bound: -14.6795541, upper bound: 14.8123530
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 75.49
Output dim: 2, lower bound: -14.6999494, upper bound: 14.7919653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 75.49
Output dim: 2, lower bound: -14.6798601, upper bound: 14.8120470
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 75.49
Output dim: 2, lower bound: -14.7919652, upper bound: 14.6798602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 75.49
Output dim: 2, lower bound: -14.7919653, upper bound: 14.6999495
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 75.49
Output dim: 2, lower bound: -14.8123529, upper bound: 14.6795541
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 75.49
Output dim: 2, lower bound: -14.7922722, upper bound: 14.6996426

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4263153, 54.4335327
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5495682, 36.5610695
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2100906, 33.2233124
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4790573, 32.4995880
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4725037, 43.4787064
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2174683, 38.2360992
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5463562, 44.5571976
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7576523, 41.7757874
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1565475, 47.1695175
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8321838, 35.8334198
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6571198, 50.6401138
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5085068, 49.5142441
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1289215, 44.0924988
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5951538, 49.5879440
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5580978, 48.5628357
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6404305, 40.6197739
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9650040, 29.9624367
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1659088, 29.1619186
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4554291, 40.4543915
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5314026, 38.5239716
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2807846, 35.2767029
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8590317, 32.8577118
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4237747, 34.4174957
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8184052, 43.7924576
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7775040, 37.7800980
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6663971, 36.6633453
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6844864, 37.6851654
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2535477, 57.2571716
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3648453, 42.3473053
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9667587, 43.9590912
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5934143, 45.5797577
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2594528, 55.2337494
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2203217, 53.2166443
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5913315, 45.5926285
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9062881, 36.9150162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1653

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6788040, upper bound: 14.8108316
time: 35.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6780333, upper bound: 14.8116010
time: 46.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4263306, 54.4335251
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5504379, 36.5601997
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2115707, 33.2218323
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4806824, 32.4979630
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4735107, 43.4776917
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2205200, 38.2330399
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5513000, 44.5522461
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7606125, 41.7728424
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1585464, 47.1675110
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8321609, 35.8334503
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6542969, 50.6429367
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5083542, 49.5143967
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1237946, 44.0976257
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5937042, 49.5893936
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5579758, 48.5629578
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6397133, 40.6204872
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9636765, 29.9637642
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1663589, 29.1614685
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4546356, 40.4551697
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5314484, 38.5239182
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2800903, 35.2774010
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8588562, 32.8578873
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4234467, 34.4178238
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8177948, 43.7930603
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7780762, 37.7795296
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6663513, 36.6633911
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6843262, 37.6853256
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2535934, 57.2571411
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3645248, 42.3476257
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9655685, 43.9602814
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5930176, 45.5801620
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2532883, 55.2399368
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2220612, 53.2149048
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5915451, 45.5924225
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9098053, 36.9114990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1653

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6791082, upper bound: 14.8105261
time: 19.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6783370, upper bound: 14.8112942
time: 32.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4343414, 54.4263153
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5618973, 36.5495720
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2239532, 33.2100906
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5001068, 32.4790573
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4790955, 43.4725037
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2362518, 38.2174606
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5571899, 44.5463943
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7790451, 41.7576599
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1716232, 47.1565399
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8359070, 35.8321877
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6387787, 50.6571198
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5183029, 49.5085068
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.0924911, 44.1315460
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5879364, 49.5976791
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5684280, 48.5580902
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6197701, 40.6400757
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9643936, 29.9650040
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1618195, 29.1659088
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4584503, 40.4554214
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5239716, 38.5362015
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2780685, 35.2807846
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8581009, 32.8590279
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4175339, 34.4237785
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7924500, 43.8187790
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7806091, 37.7775040
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6637650, 36.6663971
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6867981, 37.6844864
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2571793, 57.2536545
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3473129, 42.3665009
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9590912, 43.9668884
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5797577, 45.5947342
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2337418, 55.2599869
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2166290, 53.2207260
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5926285, 45.5913696
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9155426, 36.9062881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1653

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8116009, upper bound: 14.6780334
time: 46.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8108315, upper bound: 14.6788041
time: 20.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 69.41 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 2, lower bound: -14.6788040, upper bound: 14.8108316
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 2, lower bound: -14.6780333, upper bound: 14.8116010
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 2, lower bound: -14.6791082, upper bound: 14.8105261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 2, lower bound: -14.6783370, upper bound: 14.8112942
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 2, lower bound: -14.8116009, upper bound: 14.6780334
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 69.41
Output dim: 2, lower bound: -14.8108315, upper bound: 14.6788041

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4265976, 54.4348602
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5599976, 36.5725899
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2277985, 33.2438698
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4688797, 32.4930153
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4763412, 43.4826279
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2147522, 38.2379074
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5112000, 44.5266418
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7561188, 41.7741776
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1790619, 47.1959763
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8000183, 35.7901001
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5507812, 50.5140991
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4596634, 49.4567261
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1273880, 44.0910034
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5905914, 49.5840530
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4833069, 48.4726486
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6415787, 40.6209030
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9437027, 29.9308662
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1729584, 29.1649399
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4089203, 40.3956757
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5409546, 38.5342712
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2772446, 35.2662926
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8531189, 32.8498993
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4147491, 34.4044876
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8262482, 43.8003769
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7612457, 37.7659302
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6795502, 36.6735229
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6504135, 37.6402588
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2357178, 57.2419052
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3455429, 42.3306885
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9482346, 43.9432297
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5559998, 45.5485382
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2506943, 55.2254410
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2001114, 53.1984177
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5887222, 45.5900497
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8974457, 36.9074173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6781375, upper bound: 14.8072198
time: 34.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6751977, upper bound: 14.8101672
time: 62.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4276352, 54.4338303
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5610962, 36.5714912
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2306519, 33.2410316
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4724960, 32.4893990
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4764328, 43.4825363
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2192688, 38.2333832
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5158081, 44.5220337
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7560425, 41.7742462
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1829987, 47.1920471
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7888718, 35.8012428
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5311127, 50.5337830
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4509964, 49.4654007
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1274185, 44.0909653
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5912628, 49.5833817
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4678955, 48.4880600
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6415558, 40.6209183
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9334335, 29.9411392
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1689301, 29.1689682
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.3967133, 40.4078979
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5417175, 38.5335159
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2703705, 35.2731705
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8512192, 32.8517990
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4107666, 34.4084702
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8263092, 43.8003235
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7633286, 37.7638512
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6765518, 36.6765213
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6395798, 37.6510925
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2382660, 57.2393646
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3482132, 42.3280182
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9509048, 43.9405594
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5621796, 45.5423508
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2511673, 55.2249603
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2020798, 53.1964417
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5887833, 45.5899887
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8986816, 36.9061813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6773678, upper bound: 14.8079899
time: 32.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6744268, upper bound: 14.8109368
time: 32.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4266129, 54.4348526
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5608673, 36.5717201
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2292786, 33.2423897
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4705048, 32.4913979
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4773636, 43.4816208
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2178040, 38.2348404
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5161438, 44.5216904
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7590637, 41.7712326
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1810760, 47.1939697
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7999878, 35.7901268
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5479584, 50.5169220
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4595108, 49.4568787
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1222610, 44.0961304
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5891266, 49.5855026
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4831848, 48.4727707
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6408615, 40.6216164
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9423752, 29.9321938
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1734085, 29.1644936
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4081573, 40.3964539
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5410004, 38.5342255
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2765503, 35.2669907
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8529434, 32.8500748
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4144211, 34.4048157
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8256531, 43.8009796
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7618179, 37.7653618
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6795044, 36.6735611
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6502533, 37.6404152
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2357635, 57.2418747
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3452377, 42.3310013
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9470444, 43.9444199
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5555878, 45.5489426
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2444992, 55.2316284
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2018509, 53.1966782
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5889359, 45.5898361
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9009628, 36.9039001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6784430, upper bound: 14.8069140
time: 43.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6755022, upper bound: 14.8098615
time: 39.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4276352, 54.4338226
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5619659, 36.5706215
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2321320, 33.2395515
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4741211, 32.4877815
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4774551, 43.4815216
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2223358, 38.2303238
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5207520, 44.5170746
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7590027, 41.7713013
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1850128, 47.1900406
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7888489, 35.8012695
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5282745, 50.5366058
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4508438, 49.4655457
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1222916, 44.0960999
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5897980, 49.5848312
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4677734, 48.4881897
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6408463, 40.6216354
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9321060, 29.9424667
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1693802, 29.1685219
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.3959198, 40.4086761
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5417480, 38.5334625
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2696762, 35.2738647
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8510513, 32.8519745
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4104385, 34.4088020
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8256989, 43.8009262
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7639008, 37.7632828
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6765137, 36.6765594
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6394196, 37.6512527
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2383118, 57.2393265
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3479080, 42.3283310
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9497147, 43.9417572
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5617828, 45.5427551
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2449875, 55.2311478
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2038193, 53.1947021
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5889969, 45.5897751
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9021988, 36.9026642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6776717, upper bound: 14.8076828
time: 35.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6747310, upper bound: 14.8106296
time: 17.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4346237, 54.4276276
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5723114, 36.5611000
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2416687, 33.2306480
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4899292, 32.4724998
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4829483, 43.4764404
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2335510, 38.2192764
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5220337, 44.5158386
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7774963, 41.7560501
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1941528, 47.1829987
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8037415, 35.7888756
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5324554, 50.5311050
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4694595, 49.4509964
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.0909653, 44.1300507
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5833893, 49.5937881
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4936523, 48.4678879
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6209183, 40.6412048
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9430923, 29.9334297
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1688690, 29.1689301
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4119720, 40.3967056
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5335083, 38.5465012
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2745285, 35.2703705
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8521957, 32.8512230
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4085083, 34.4107666
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8003235, 43.8266983
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7643585, 37.7633286
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6769180, 36.6765518
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6527252, 37.6395798
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2393799, 57.2383881
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3280106, 42.3498764
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9405594, 43.9510269
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5423584, 45.5635147
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2249680, 55.2516785
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.1964493, 53.2024994
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5899734, 45.5887833
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9067001, 36.8986816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8109367, upper bound: 14.6744269
time: 56.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6990045, upper bound: 14.6773679
time: 33.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4356461, 54.4265976
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5734253, 36.5599937
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2445068, 33.2278023
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4935455, 32.4688835
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4830399, 43.4763412
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2380676, 38.2147522
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5266418, 44.5112305
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7774353, 41.7561188
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1980743, 47.1790695
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7925949, 35.8000183
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5127716, 50.5507812
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4607925, 49.4596634
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.0909958, 44.1300125
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5840607, 49.5931168
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4782257, 48.4833069
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6209030, 40.6412201
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9328232, 29.9437027
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1648407, 29.1729584
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.3997345, 40.4089279
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5342712, 38.5457382
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2676544, 35.2772484
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8502960, 32.8531189
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4045181, 34.4147530
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8003693, 43.8266449
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7664337, 37.7612457
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6739197, 36.6795502
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6418839, 37.6504135
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2419128, 57.2358475
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3306808, 42.3472137
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9432297, 43.9483643
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5485382, 45.5573273
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2254410, 55.2512054
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.1984177, 53.2005234
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5900345, 45.5887222
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9079361, 36.8974457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1553

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7653097, upper bound: 14.6751977
time: 37.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8072197, upper bound: 14.6781376
time: 38.15 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 77.70 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6781375, upper bound: 14.8072198
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6751977, upper bound: 14.8101672
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6773678, upper bound: 14.8079899
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6744268, upper bound: 14.8109368
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6784430, upper bound: 14.8069140
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6755022, upper bound: 14.8098615
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6776717, upper bound: 14.8076828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6747310, upper bound: 14.8106296
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.8109367, upper bound: 14.6744269
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.6990045, upper bound: 14.6773679
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.7653097, upper bound: 14.6751977
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 77.70
Output dim: 2, lower bound: -14.8072197, upper bound: 14.6781376

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4147491, 54.4244308
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5575790, 36.5710526
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2241592, 33.2412262
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4700470, 32.4942932
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4762344, 43.4825287
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2164383, 38.2401123
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5135193, 44.5283203
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7580338, 41.7764740
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1733704, 47.1911011
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7928009, 35.7815437
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5302429, 50.4908752
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4610672, 49.4581909
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1263885, 44.0895691
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5868835, 49.5802917
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4810257, 48.4701767
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6463852, 40.6269989
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9432449, 29.9303131
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1682434, 29.1595573
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4091415, 40.3958359
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5443268, 38.5369034
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2748413, 35.2638588
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8530960, 32.8498993
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4140778, 34.4036903
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8251953, 43.7991943
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7629242, 37.7681732
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6791153, 36.6728973
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6503792, 37.6402130
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2475586, 57.2519608
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3641281, 42.3462448
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9503860, 43.9452972
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5505524, 45.5435333
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2444611, 55.2194672
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2005615, 53.1977921
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5911255, 45.5921478
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9006271, 36.9097900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1691

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6732697, upper bound: 14.8043776
time: 36.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6694280, upper bound: 14.8082289
time: 51.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4157867, 54.4234009
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5586853, 36.5699463
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2269974, 33.2383804
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4736633, 32.4906769
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4763260, 43.4824371
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2209702, 38.2355881
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5181427, 44.5237045
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7579727, 41.7765503
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1772919, 47.1871719
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7816544, 35.7926903
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5105591, 50.5105591
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4524002, 49.4668579
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1264343, 44.0895386
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5875549, 49.5796127
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4656143, 48.4855957
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6463623, 40.6270142
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9329758, 29.9405861
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1642151, 29.1635857
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.3969345, 40.4080582
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5450897, 38.5361481
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2679672, 35.2707367
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8512039, 32.8517952
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4100876, 34.4076767
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8252411, 43.7991409
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7650070, 37.7660904
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6761169, 36.6758957
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6395454, 37.6510468
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2501068, 57.2494202
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3667984, 42.3435745
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9530563, 43.9426270
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5567474, 45.5373383
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2449341, 55.2189865
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2025452, 53.1958160
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5911865, 45.5920868
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9018631, 36.9085541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1691

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6724989, upper bound: 14.8051413
time: 35.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6686605, upper bound: 14.8089929
time: 32.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4147644, 54.4244232
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5584488, 36.5701828
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2256393, 33.2397461
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4716644, 32.4926682
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4772415, 43.4815140
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2195053, 38.2370529
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5184784, 44.5233612
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7609940, 41.7735291
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1753693, 47.1890869
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7927704, 35.7815742
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5274353, 50.4936981
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4609146, 49.4583435
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1212616, 44.0947037
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5854340, 49.5817413
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4809036, 48.4702988
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6456680, 40.6277084
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9419174, 29.9316406
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1686935, 29.1591072
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4083786, 40.3966141
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5443726, 38.5368576
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2741470, 35.2645569
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8529205, 32.8500748
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4137497, 34.4040222
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8245850, 43.7997971
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7634964, 37.7676010
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6790695, 36.6729431
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6502190, 37.6403732
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2476044, 57.2519226
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3638077, 42.3465652
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9491959, 43.9464874
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5501556, 45.5439377
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2382812, 55.2256546
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2023010, 53.1960526
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5913391, 45.5919342
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9041367, 36.9062729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1691

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6736067, upper bound: 14.8041013
time: 36.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6697012, upper bound: 14.8078866
time: 33.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4157867, 54.4233932
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5595551, 36.5690765
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2284775, 33.2369003
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4752808, 32.4890518
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4773331, 43.4814148
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2240219, 38.2325287
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5231018, 44.5187531
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7609177, 41.7735977
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1792908, 47.1851654
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7816315, 35.7927208
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5077515, 50.5133820
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4522476, 49.4670105
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1212921, 44.0946655
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5861053, 49.5810623
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4654922, 48.4857178
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6456528, 40.6277275
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9316483, 29.9419136
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1646652, 29.1631393
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.3961411, 40.4088364
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5451202, 38.5360947
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2672729, 35.2714310
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8510284, 32.8519669
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4097595, 34.4080048
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8246460, 43.7997437
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7655792, 37.7655220
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6760712, 36.6759415
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6393852, 37.6512070
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2501373, 57.2493820
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3664780, 42.3438950
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9518509, 43.9438171
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5563507, 45.5377502
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2387543, 55.2251740
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2042847, 53.1940765
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5914001, 45.5918732
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9053726, 36.9050369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1691

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6728360, upper bound: 14.8048644
time: 35.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6689339, upper bound: 14.8086513
time: 36.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4241791, 54.4157867
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5707779, 36.5586853
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2390213, 33.2269974
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4912109, 32.4736557
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4828262, 43.4763184
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2357407, 38.2209625
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5236969, 44.5181808
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7797928, 41.7579727
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1892700, 47.1772919
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7951736, 35.7816582
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5092163, 50.5105667
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4709244, 49.4523926
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.0895386, 44.1290512
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5796204, 49.5900879
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4911880, 48.4656143
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6270142, 40.6459961
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9425430, 29.9329796
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1634827, 29.1642189
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4121323, 40.3969269
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5361481, 38.5498734
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2720795, 35.2679672
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8521805, 32.8512001
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4077225, 34.4100952
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7991333, 43.8256378
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7665939, 37.7650070
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6762848, 36.6761169
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6526833, 37.6395416
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2494354, 57.2502136
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3435822, 42.3684464
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9426346, 43.9531708
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5373383, 45.5580826
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2189941, 55.2454605
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.1958313, 53.2029495
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5920868, 45.5912018
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9090805, 36.9018631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1691

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8089928, upper bound: 14.6686605
time: 40.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8051413, upper bound: 14.6724990
time: 41.13 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 83.68 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.6732697, upper bound: 14.8043776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.6694280, upper bound: 14.8082289
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.6724989, upper bound: 14.8051413
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.6686605, upper bound: 14.8089929
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.6736067, upper bound: 14.8041013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.6697012, upper bound: 14.8078866
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.6728360, upper bound: 14.8048644
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.6689339, upper bound: 14.8086513
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.8089928, upper bound: 14.6686605
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 83.68
Output dim: 2, lower bound: -14.8051413, upper bound: 14.6724990

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4147339, 54.4231262
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5592880, 36.5693893
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2270432, 33.2381668
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4747696, 32.4857559
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4752274, 43.4799652
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2218399, 38.2349014
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5205841, 44.5213013
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7574081, 41.7763901
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1764832, 47.1846619
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7798080, 35.7945557
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5003357, 50.5040665
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4445419, 49.4645004
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1255722, 44.0887146
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5858612, 49.5781784
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4554062, 48.4816589
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6462936, 40.6269798
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9235458, 29.9393997
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1620598, 29.1631203
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.3869476, 40.4064789
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5445099, 38.5347977
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2618256, 35.2701416
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8508797, 32.8516846
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4080734, 34.4075851
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8232880, 43.7980881
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7635651, 37.7655029
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6748657, 36.6756821
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6319046, 37.6497459
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2488556, 57.2420044
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3643799, 42.3304443
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9521027, 43.9400024
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5567169, 45.5373230
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2444916, 55.2197952
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2017212, 53.1907578
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5911255, 45.5920029
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9023972, 36.9067078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1706

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6677020, upper bound: 14.8054673
time: 32.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6647751, upper bound: 14.8080832
time: 36.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4239197, 54.4147339
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5702133, 36.5592880
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2388077, 33.2270432
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4862823, 32.4747696
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4803543, 43.4752274
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2350693, 38.2218475
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5213013, 44.5206299
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7796402, 41.7574081
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1867523, 47.1764832
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.7970505, 35.7798119
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.5027313, 50.5003357
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.4685593, 49.4445343
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.0887222, 44.1281967
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5781708, 49.5884018
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.4872665, 48.4554062
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6269836, 40.6459351
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9413528, 29.9235420
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1630287, 29.1620560
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4105377, 40.3869476
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5347900, 38.5493011
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2714996, 35.2618256
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8520851, 32.8508797
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4076309, 34.4080734
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7980804, 43.8236847
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7660141, 37.7635651
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6760864, 36.6748657
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6513748, 37.6319046
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2420044, 57.2489700
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3304367, 42.3660431
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9399948, 43.9522247
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5373230, 45.5580444
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2198029, 55.2450104
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.1907654, 53.2021255
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5920105, 45.5911484
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9072342, 36.9023972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=180, inp2_unstable=180, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1706

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8080831, upper bound: 14.6647752
time: 36.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.8054672, upper bound: 14.6677020
time: 32.60 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 71.07 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 71.07
Output dim: 2, lower bound: -14.6677020, upper bound: 14.8054673
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 71.07
Output dim: 2, lower bound: -14.6647751, upper bound: 14.8080832
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 71.07
Output dim: 2, lower bound: -14.8080831, upper bound: 14.6647752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 71.07
Output dim: 2, lower bound: -14.8054672, upper bound: 14.6677020

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 49.96 + 2074.54 = 2124.49 seconds
