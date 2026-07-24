## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.26 + 47.91 = 50.16 seconds
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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7772129, upper bound: 14.8220449
time: 38.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8220448, upper bound: 14.7772129
time: 35.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 73.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 73.77
Output dim: 2, lower bound: -14.7772129, upper bound: 14.8220449
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 73.77
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

Time for backsubstitution: 1.75 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7702758, upper bound: 14.7916132
time: 35.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7467707, upper bound: 14.8151219
time: 54.69 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8151218, upper bound: 14.7467708
time: 31.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7916131, upper bound: 14.7702759
time: 44.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 78.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 78.11
Output dim: 2, lower bound: -14.7702758, upper bound: 14.7916132
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 78.11
Output dim: 2, lower bound: -14.7467707, upper bound: 14.8151219
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 78.11
Output dim: 2, lower bound: -14.8151218, upper bound: 14.7467708
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 78.11
Output dim: 2, lower bound: -14.7916131, upper bound: 14.7702759

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4371490, 54.4331894
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5586853, 36.5560036
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2280731, 33.2277260
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5096588, 32.5115433
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4833374, 43.4851379
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2433929, 38.2446976
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5400848, 44.5407944
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7665405, 41.7647171
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1804657, 47.1790314
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8373871, 35.8395615
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6859283, 50.6867523
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5226822, 49.5181732
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1517410, 44.1521530
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6267624, 49.6263885
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5746460, 48.5661469
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6393661, 40.6370850
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9846497, 29.9803658
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1610641, 29.1648254
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4770508, 40.4739838
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5326920, 38.5373917
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2905884, 35.2881165
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8625336, 32.8623123
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4271545, 34.4263306
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8173141, 43.8229904
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8109665, 37.8139420
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6660156, 36.6650162
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6893005, 37.6847343
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2684479, 57.2682114
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3539047, 42.3562927
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9651337, 43.9618073
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5879745, 45.5902100
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2421417, 55.2367783
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2233658, 53.2235413
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5923538, 45.5915604
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8998566, 36.9008942

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7244398, upper bound: 14.7897995
time: 21.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7684816, upper bound: 14.7457829
time: 17.66 seconds

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

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7009470, upper bound: 14.8133181
time: 32.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7449636, upper bound: 14.7692951
time: 46.49 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7692951, upper bound: 14.7449637
time: 42.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8133180, upper bound: 14.7009470
time: 38.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4331970, 54.4363480
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5560074, 36.5578575
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2277222, 33.2274361
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5115433, 32.5091400
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4851379, 43.4829483
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2446899, 38.2432327
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5407562, 44.5400925
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7647095, 41.7632980
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1790314, 47.1783676
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8395615, 35.8348846
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6867523, 50.6872559
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5181656, 49.5185852
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1495285, 44.1517487
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6238632, 49.6267548
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5661469, 48.5690536
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6374435, 40.6393661
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9803619, 29.9826927
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1648178, 29.1611633
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4739838, 40.4729843
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5326004, 38.5326843
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2881165, 35.2892494
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8623123, 32.8621292
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4263306, 34.4271126
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8226089, 43.8173218
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8139420, 37.8104706
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6650162, 36.6656036
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6847305, 37.6876831
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2680969, 57.2684326
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3546219, 42.3539047
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9616928, 43.9651337
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5888901, 45.5879745
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2362518, 55.2421494
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2231369, 53.2233658
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5915604, 45.5923462
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9008942, 36.8993301

Time for backsubstitution: 1.77 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7457828, upper bound: 14.7684817
time: 39.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7897995, upper bound: 14.7244399
time: 36.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 77.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 77.20
Output dim: 2, lower bound: -14.7244398, upper bound: 14.7897995
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 77.20
Output dim: 2, lower bound: -14.7684816, upper bound: 14.7457829
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 77.20
Output dim: 2, lower bound: -14.7009470, upper bound: 14.8133181
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 77.20
Output dim: 2, lower bound: -14.7449636, upper bound: 14.7692951
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 77.20
Output dim: 2, lower bound: -14.7692951, upper bound: 14.7449637
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 77.20
Output dim: 2, lower bound: -14.8133180, upper bound: 14.7009470
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 77.20
Output dim: 2, lower bound: -14.7457828, upper bound: 14.7684817
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 77.20
Output dim: 2, lower bound: -14.7897995, upper bound: 14.7244399

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4321747, 54.4303513
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5526428, 36.5542336
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2187500, 33.2244606
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4897385, 32.5054169
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4802856, 43.4841461
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2296143, 38.2405548
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5371094, 44.5428314
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7600708, 41.7600174
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1721039, 47.1765366
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8361816, 35.8333359
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6781616, 50.6636810
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5185852, 49.5110168
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1449203, 44.1340103
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6238327, 49.6249008
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5751724, 48.5595627
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6361084, 40.6174049
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9832458, 29.9714165
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1576691, 29.1520309
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4721756, 40.4601746
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5300064, 38.5341873
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2905579, 35.2802963
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8627548, 32.8583794
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4261475, 34.4181633
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8091583, 43.8007584
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8097839, 37.8082809
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6681747, 36.6630630
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6932144, 37.6784210
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2512360, 57.2625580
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3465881, 42.3522263
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9505539, 43.9564819
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5870209, 45.5892792
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2414627, 55.2345352
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2225571, 53.2229919
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5909805, 45.5922623
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.8992310, 36.9023972

Time for backsubstitution: 1.71 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7240985, upper bound: 14.7897950
time: 46.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7244211, upper bound: 14.7895027
time: 42.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4343109, 54.4282303
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5569153, 36.5499611
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2248077, 33.2183952
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5035324, 32.4916229
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4823456, 43.4820862
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2392426, 38.2309265
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5421295, 44.5378189
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7618561, 41.7582397
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1779938, 47.1706543
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8311615, 35.8383598
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6628571, 50.6789856
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5155334, 49.5140762
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1336136, 44.1453323
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6252670, 49.6234589
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5680618, 48.5666809
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6196823, 40.6338310
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9757080, 29.9789619
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1482697, 29.1614342
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4632492, 40.4690933
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5294724, 38.5347214
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2827682, 35.2880859
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8585968, 32.8625374
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4189911, 34.4253235
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7950897, 43.8148346
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8053131, 37.8127518
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6640625, 36.6671753
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6829910, 37.6886444
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2627716, 57.2510147
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3498383, 42.3489761
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9598007, 43.9472275
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5870514, 45.5892715
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2399063, 55.2360992
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2228165, 53.2227325
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5930557, 45.5901871
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9013596, 36.9002686

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7240985, upper bound: 14.7457766
time: 37.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7684759, upper bound: 14.7454702
time: 34.53 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7006348, upper bound: 14.8133131
time: 42.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7009412, upper bound: 14.8130060
time: 36.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4297638, 54.4319763
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5522003, 36.5538445
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2216339, 33.2209282
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5013504, 32.4932861
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4814453, 43.4825745
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2369537, 38.2330627
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5424042, 44.5375061
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7533264, 41.7635345
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1739197, 47.1726074
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8282242, 35.8388023
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6634521, 50.6797180
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5077209, 49.5178146
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1418076, 44.1344833
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6269150, 49.6192856
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5563278, 48.5728226
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6204147, 40.6334686
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9701462, 29.9825668
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1482697, 29.1615334
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4561081, 40.4721756
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5322037, 38.5272141
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2791901, 35.2903175
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8570328, 32.8637009
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4182739, 34.4259872
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8030701, 43.8064957
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8040390, 37.8135262
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6630096, 36.6678162
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6753693, 37.6946449
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2656097, 57.2480545
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3586960, 42.3384705
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9645767, 43.9423447
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5936432, 45.5813522
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2454910, 55.2300034
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2255173, 53.2196198
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5932083, 45.5900192
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9005127, 36.9005890

Time for backsubstitution: 1.75 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7446683, upper bound: 14.7692813
time: 41.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7449596, upper bound: 14.7689553
time: 39.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4327698, 54.4297562
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5546722, 36.5521965
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2215729, 33.2216377
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4938126, 32.5013504
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4829712, 43.4814606
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2332153, 38.2369537
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5375061, 44.5424347
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7667847, 41.7533188
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1747131, 47.1739197
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8412933, 35.8282280
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6783905, 50.6634521
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5218811, 49.5077057
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1344833, 44.1444473
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6192856, 49.6294403
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5784073, 48.5563202
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6334686, 40.6200485
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9845200, 29.9701462
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1614380, 29.1482658
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4762344, 40.4561081
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5272141, 38.5369873
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2916718, 35.2791824
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8640976, 32.8570328
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4260406, 34.4182739
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8064880, 43.8034363
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8140335, 37.8040352
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6682281, 36.6630096
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6962662, 37.6753693
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2480621, 57.2657394
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3384705, 42.3603439
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9423447, 43.9646988
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5813446, 45.5949631
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2299881, 55.2460098
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2196121, 53.2259369
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5900345, 45.5932159
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9011154, 36.9005127

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7689552, upper bound: 14.7449597
time: 32.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7692812, upper bound: 14.7446684
time: 69.94 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8130059, upper bound: 14.7009412
time: 39.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8133130, upper bound: 14.7006349
time: 42.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4282379, 54.4335022
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5499573, 36.5560799
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2183990, 33.2241707
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4916229, 32.5030136
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4820862, 43.4819489
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2309265, 38.2390900
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5377960, 44.5421219
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7582397, 41.7586060
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1706543, 47.1758728
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8383560, 35.8286705
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6789856, 50.6641846
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5140686, 49.5114441
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1426926, 44.1336060
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6209335, 49.6252670
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5666733, 48.5624619
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6341934, 40.6196823
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9789581, 29.9737511
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1614380, 29.1483650
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4690933, 40.4591904
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5299301, 38.5294800
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2880859, 35.2814140
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8625336, 32.8581963
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4253235, 34.4189377
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8144531, 43.7950897
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8127518, 37.8048096
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6671753, 36.6636505
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6886444, 37.6813660
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2508850, 57.2627792
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3473129, 42.3498383
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9471054, 43.9598083
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5879517, 45.5870438
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2355881, 55.2399063
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2223129, 53.2228165
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5901566, 45.5930557
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9002686, 36.9008331

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7454701, upper bound: 14.7684760
time: 37.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7009412, upper bound: 14.7681662
time: 53.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4303436, 54.4313812
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5542297, 36.5518074
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2244568, 33.2181053
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.5054169, 32.4892197
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4841309, 43.4798889
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2405548, 38.2294617
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5428009, 44.5371094
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7600098, 41.7568283
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1765442, 47.1699829
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8333359, 35.8336945
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6636810, 50.6794891
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5110168, 49.5145035
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1313705, 44.1449203
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.6223679, 49.6238251
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5595627, 48.5695801
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6177673, 40.6361084
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9714203, 29.9812965
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1520386, 29.1577644
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4601822, 40.4681015
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5294113, 38.5300140
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2802963, 35.2892075
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8583832, 32.8623543
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4181671, 34.4260979
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8003845, 43.8091660
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.8082809, 37.8092804
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6630630, 36.6677704
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6784210, 37.6915894
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2624359, 57.2512360
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3505783, 42.3465881
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9563675, 43.9505539
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5879517, 45.5870361
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2340164, 55.2414780
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2225723, 53.2225647
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5922623, 45.5909653
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9023972, 36.8987045

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7895027, upper bound: 14.7244212
time: 36.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7897949, upper bound: 14.7240986
time: 39.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 77.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7240985, upper bound: 14.7897950
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7244211, upper bound: 14.7895027
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7240985, upper bound: 14.7457766
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7684759, upper bound: 14.7454702
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7006348, upper bound: 14.8133131
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7009412, upper bound: 14.8130060
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7446683, upper bound: 14.7692813
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7449596, upper bound: 14.7689553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7689552, upper bound: 14.7449597
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7692812, upper bound: 14.7446684
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.8130059, upper bound: 14.7009412
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.8133130, upper bound: 14.7006349
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7454701, upper bound: 14.7684760
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7009412, upper bound: 14.7681662
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7895027, upper bound: 14.7244212
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 77.27
Output dim: 2, lower bound: -14.7897949, upper bound: 14.7240986

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4321289, 54.4303131
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5550461, 36.5575066
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2138138, 33.2210083
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4806671, 32.4979782
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4733582, 43.4782257
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2201233, 38.2341156
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5462036, 44.5568771
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7686310, 41.7715149
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1614456, 47.1679001
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8363647, 35.8334923
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6570740, 50.6397705
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5195465, 49.5118256
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1216507, 44.1056061
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5942535, 49.5938721
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5742874, 48.5585403
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6402397, 40.6208229
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9725952, 29.9594345
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1660080, 29.1608200
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4651566, 40.4523926
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5297089, 38.5339432
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2858353, 35.2748795
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8614044, 32.8568573
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4245529, 34.4162407
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8108978, 43.8018723
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7813110, 37.7803802
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6674728, 36.6623154
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6941490, 37.6792030
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2520981, 57.2634506
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3585052, 42.3638229
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9639587, 43.9686966
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5875244, 45.5893631
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2545242, 55.2414017
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2176208, 53.2197952
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5912018, 45.5927124
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9072723, 36.9139519

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7231288, upper bound: 14.7687494
time: 33.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7030321, upper bound: 14.7888326
time: 51.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4321442, 54.4303055
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5559158, 36.5566368
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2152939, 33.2195282
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4822998, 32.4963531
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4743652, 43.4772186
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2231750, 38.2310562
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5511475, 44.5519257
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7715759, 41.7685699
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1634598, 47.1658936
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8363342, 35.8335190
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6542511, 50.6425934
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5193939, 49.5119781
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1165085, 44.1107330
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5927887, 49.5953217
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5741653, 48.5586624
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6395226, 40.6215363
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9712677, 29.9607620
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1664658, 29.1603737
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4643936, 40.4531708
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5297699, 38.5338898
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2851410, 35.2755737
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8612289, 32.8570328
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4242249, 34.4165726
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8102875, 43.8024750
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7818832, 37.7798119
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6674271, 36.6623535
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6939888, 37.6793594
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2521286, 57.2634048
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3581848, 42.3641434
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9627686, 43.9698868
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5871277, 45.5897675
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2483444, 55.2475891
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2193604, 53.2180557
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5914154, 45.5924988
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9107895, 36.9104385

Time for backsubstitution: 1.86 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7234544, upper bound: 14.7684592
time: 24.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7033545, upper bound: 14.7885407
time: 38.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4342651, 54.4281921
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5593109, 36.5532341
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2198715, 33.2149429
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4944763, 32.4841843
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4754181, 43.4761734
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2297516, 38.2244873
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5512085, 44.5518646
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7704163, 41.7697372
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1673355, 47.1620102
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8313446, 35.8385162
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6417694, 50.6550827
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5164795, 49.5148849
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1103287, 44.1169205
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5956879, 49.5924301
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5671768, 48.5656586
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6238136, 40.6372490
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9650497, 29.9669800
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1566086, 29.1702232
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4562454, 40.4613037
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5291901, 38.5344696
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2780457, 35.2826691
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8572540, 32.8610153
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4173965, 34.4234009
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7967987, 43.8159485
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7768402, 37.7848511
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6633530, 36.6664276
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6839256, 37.6894264
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2636337, 57.2518997
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3617554, 42.3605728
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9732208, 43.9594498
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5875244, 45.5893555
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2529678, 55.2429657
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2178802, 53.2195358
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5932770, 45.5906219
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9094009, 36.9118233

Time for backsubstitution: 1.77 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7671855, upper bound: 14.7247373
time: 48.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7470934, upper bound: 14.7448107
time: 38.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4342651, 54.4281845
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5601807, 36.5523643
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2213516, 33.2134628
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4960938, 32.4825592
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4764404, 43.4751587
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2328033, 38.2214279
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5561829, 44.5469131
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7733459, 41.7667923
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1693497, 47.1600037
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8313141, 35.8385391
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6389465, 50.6579056
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5163269, 49.5150375
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1052017, 44.1220551
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5942383, 49.5938797
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5670395, 48.5657806
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6230965, 40.6379623
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9637222, 29.9683075
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1570663, 29.1697731
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4554672, 40.4620819
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5292358, 38.5344238
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2773514, 35.2833633
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8570709, 32.8611908
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4170685, 34.4237289
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7962189, 43.8165512
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7774124, 37.7842827
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6633148, 36.6664658
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6837654, 37.6895828
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2636642, 57.2518616
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3614349, 42.3608932
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9720306, 43.9606400
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5871277, 45.5897598
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2467728, 55.2491531
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2196198, 53.2177963
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5934906, 45.5904160
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9129181, 36.9083099

Time for backsubstitution: 1.77 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7674929, upper bound: 14.7244317
time: 33.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7474005, upper bound: 14.7445045
time: 38.55 seconds

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

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6996425, upper bound: 14.7922723
time: 45.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6795541, upper bound: 14.8123530
time: 32.26 seconds

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

Time for backsubstitution: 1.84 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6999494, upper bound: 14.7919653
time: 39.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6798601, upper bound: 14.8120470
time: 34.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4297180, 54.4319534
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5546036, 36.5571175
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2167053, 33.2174759
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4922791, 32.4858475
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4745178, 43.4766693
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2274475, 38.2266235
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5515137, 44.5515594
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7618713, 41.7750320
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1632767, 47.1639862
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8284073, 35.8389511
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6423645, 50.6558151
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5086670, 49.5186310
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1185532, 44.1060791
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5973053, 49.5882568
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5554276, 48.5718079
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6245384, 40.6368866
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9594955, 29.9705887
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1566086, 29.1703224
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4491043, 40.4643860
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5319061, 38.5269699
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2744598, 35.2848892
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8556900, 32.8621750
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4166794, 34.4240570
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8047791, 43.8076096
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7755661, 37.7856369
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6623001, 36.6670761
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6763039, 37.6954269
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2664719, 57.2489395
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3705978, 42.3500748
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9779739, 43.9545593
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5941162, 45.5814362
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2585526, 55.2368698
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2205963, 53.2164154
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5934296, 45.5904617
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9085541, 36.9121437

Time for backsubstitution: 1.85 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7436840, upper bound: 14.7482388
time: 49.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7235871, upper bound: 14.7683252
time: 48.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4297180, 54.4319458
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5554733, 36.5562477
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2181854, 33.2159958
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4939117, 32.4842300
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4755402, 43.4756546
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2305145, 38.2235718
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5564575, 44.5466003
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7648010, 41.7720871
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1652908, 47.1619797
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8283768, 35.8389778
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6395416, 50.6586380
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5085144, 49.5187836
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1134109, 44.1112137
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5958710, 49.5897064
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5553055, 48.5719299
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6238289, 40.6375961
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9581680, 29.9719124
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1570511, 29.1698761
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4483261, 40.4651642
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5319519, 38.5269165
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2737656, 35.2855835
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8555145, 32.8623505
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4163513, 34.4243851
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8041687, 43.8082123
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7761383, 37.7850647
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6622620, 36.6671219
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6761436, 37.6955833
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2665024, 57.2489014
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3702774, 42.3503876
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9767838, 43.9557495
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5937195, 45.5818405
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2523575, 55.2430573
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2223358, 53.2146759
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5936432, 45.5902481
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9120712, 36.9086304

Time for backsubstitution: 1.71 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7439767, upper bound: 14.7479138
time: 37.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7238778, upper bound: 14.7679981
time: 40.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4327240, 54.4297180
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5570755, 36.5554695
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2166290, 33.2181854
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4847412, 32.4939117
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4760437, 43.4755402
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2237244, 38.2305145
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5466003, 44.5564880
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7753296, 41.7648087
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1640701, 47.1652756
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8414688, 35.8283844
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6573029, 50.6395416
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5228577, 49.5085144
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1112137, 44.1160431
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5897064, 49.5984116
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5775223, 48.5553055
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6375923, 40.6234665
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9738617, 29.9581680
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1697769, 29.1570549
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4692307, 40.4483185
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5269165, 38.5367432
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2869492, 35.2737656
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8627472, 32.8555145
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4244461, 34.4163513
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8082123, 43.8045425
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7855606, 37.7761383
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6675186, 36.6622620
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6972084, 37.6761475
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2488937, 57.2666321
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3503876, 42.3719406
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9557495, 43.9769058
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5818481, 45.5950394
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2430496, 55.2528763
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2146759, 53.2227402
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5902557, 45.5936584
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9091568, 36.9120712

Time for backsubstitution: 1.80 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7679980, upper bound: 14.7238778
time: 36.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7479137, upper bound: 14.7439768
time: 74.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4327393, 54.4297104
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5579453, 36.5546074
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2181091, 33.2167053
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4863739, 32.4922867
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4770508, 43.4745331
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2267761, 38.2274551
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5515442, 44.5515289
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7782745, 41.7618637
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1660843, 47.1632690
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8414459, 35.8284111
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6544800, 50.6423645
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5227051, 49.5086670
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1060715, 44.1211700
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5882568, 49.5998611
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5774002, 48.5554276
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6368828, 40.6241798
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9725342, 29.9594955
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1702347, 29.1566048
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4684525, 40.4490967
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5269623, 38.5366898
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2862473, 35.2744598
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8625793, 32.8556900
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4241104, 34.4166794
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8076019, 43.8051529
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7861328, 37.7755661
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6674805, 36.6623001
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6970482, 37.6763039
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2489548, 57.2665939
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3500671, 42.3722610
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9545593, 43.9781036
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5814209, 45.5954437
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2368698, 55.2590637
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2164154, 53.2210007
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5904541, 45.5934448
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9126740, 36.9085541

Time for backsubstitution: 1.81 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7683251, upper bound: 14.7235871
time: 33.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7482387, upper bound: 14.7436841
time: 38.96 seconds

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

Time for backsubstitution: 1.80 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7919652, upper bound: 14.6798602
time: 35.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7919653, upper bound: 14.6999495
time: 33.88 seconds

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

Time for backsubstitution: 1.81 seconds

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
time: 40.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7922722, upper bound: 14.6996426
time: 34.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4281921, 54.4334869
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5523682, 36.5593529
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2134628, 33.2207184
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4825592, 32.4955750
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4751587, 43.4760437
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2214355, 38.2326584
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5468750, 44.5561676
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7667847, 41.7701111
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1600113, 47.1672516
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8385391, 35.8288193
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6579132, 50.6402817
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5150299, 49.5122681
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1194229, 44.1051941
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5913391, 49.5942383
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5657883, 48.5614548
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6383247, 40.6231003
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9683075, 29.9617729
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1697769, 29.1571541
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4620895, 40.4514008
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5296326, 38.5292358
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2833633, 35.2759857
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8611908, 32.8566742
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4237289, 34.4170074
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8161774, 43.7962036
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7842865, 37.7769165
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6664658, 36.6629105
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6895790, 37.6821442
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2517471, 57.2636719
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3592300, 42.3614426
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9605026, 43.9720230
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5884399, 45.5871201
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2486496, 55.2467804
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2173920, 53.2196198
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5904083, 45.5934906
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9083099, 36.9123917

Time for backsubstitution: 1.74 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7445044, upper bound: 14.7474006
time: 43.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7244316, upper bound: 14.7674930
time: 46.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4281921, 54.4334717
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5532379, 36.5584831
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2149429, 33.2192383
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4841766, 32.4939499
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4761658, 43.4750214
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2244873, 38.2295990
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5518494, 44.5512161
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7697449, 41.7671585
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1620255, 47.1652451
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8385086, 35.8288498
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6550751, 50.6431046
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5148773, 49.5124207
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1142960, 44.1103287
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5898895, 49.5956802
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5656509, 48.5615768
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6376076, 40.6238174
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9669800, 29.9631004
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1702194, 29.1567078
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4613113, 40.4521790
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5296783, 38.5291901
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2826691, 35.2766838
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8610153, 32.8568497
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4233932, 34.4173393
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8155670, 43.7968063
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7848511, 37.7763481
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6664276, 36.6629562
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6894264, 37.6823044
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2517776, 57.2636337
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3589249, 42.3617554
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9593124, 43.9732132
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5880127, 45.5875244
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2424545, 55.2529602
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2191315, 53.2178802
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5905914, 45.5932846
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9118195, 36.9088745

Time for backsubstitution: 1.81 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7448106, upper bound: 14.7470935
time: 33.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7247372, upper bound: 14.7671855
time: 42.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4302979, 54.4313583
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5566330, 36.5550804
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2195282, 33.2146530
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4963531, 32.4817810
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4772034, 43.4739838
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2310486, 38.2230301
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5518799, 44.5511627
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7685699, 41.7683334
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1659012, 47.1613693
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8335190, 35.8338432
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6425934, 50.6555862
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5119781, 49.5153198
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1081161, 44.1165161
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5927887, 49.5927963
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5586624, 48.5685730
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6218910, 40.6395264
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9607620, 29.9693184
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1603775, 29.1665535
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4531631, 40.4603195
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5290985, 38.5297699
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2755737, 35.2837753
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8570328, 32.8608322
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4165726, 34.4241676
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8021088, 43.8102798
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7798157, 37.7813873
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6623535, 36.6670227
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6793556, 37.6923676
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2632828, 57.2521286
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3624802, 42.3581924
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9697647, 43.9627686
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5884399, 45.5871124
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2470779, 55.2483444
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2176514, 53.2193604
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5924835, 45.5914154
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9104385, 36.9102631

Time for backsubstitution: 1.81 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7684591, upper bound: 14.7033546
time: 38.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7684591, upper bound: 14.7234544
time: 36.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4303131, 54.4313507
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5575027, 36.5542107
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2210083, 33.2131729
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4979706, 32.4801559
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4782257, 43.4729614
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2341156, 38.2199707
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5568542, 44.5462036
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7715149, 41.7653885
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1679001, 47.1593628
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8334885, 35.8338699
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6397705, 50.6584091
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5118256, 49.5154724
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1029739, 44.1216507
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5913239, 49.5942459
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5585403, 48.5686951
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6211815, 40.6402397
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9594345, 29.9706421
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1608200, 29.1661072
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4524002, 40.4610977
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5291595, 38.5297165
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2748795, 35.2844734
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8568573, 32.8610039
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4162369, 34.4244995
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8014984, 43.8108826
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7803802, 37.7808151
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6623154, 36.6670685
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6792030, 37.6925278
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2633286, 57.2520828
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3621750, 42.3585052
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9685745, 43.9639664
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5880432, 45.5875168
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2408829, 55.2545319
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2193909, 53.2176208
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5926971, 45.5912018
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9139557, 36.9067459

Time for backsubstitution: 1.72 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7888325, upper bound: 14.7030322
time: 19.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7687493, upper bound: 14.7231289
time: 36.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 57.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7231288, upper bound: 14.7687494
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7030321, upper bound: 14.7888326
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7234544, upper bound: 14.7684592
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7033545, upper bound: 14.7885407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7671855, upper bound: 14.7247373
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7470934, upper bound: 14.7448107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7674929, upper bound: 14.7244317
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7474005, upper bound: 14.7445045
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.6996425, upper bound: 14.7922723
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.6795541, upper bound: 14.8123530
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.6999494, upper bound: 14.7919653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.6798601, upper bound: 14.8120470
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7436840, upper bound: 14.7482388
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7235871, upper bound: 14.7683252
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7439767, upper bound: 14.7479138
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7238778, upper bound: 14.7679981
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7679980, upper bound: 14.7238778
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7479137, upper bound: 14.7439768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7683251, upper bound: 14.7235871
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7482387, upper bound: 14.7436841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7919652, upper bound: 14.6798602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7919653, upper bound: 14.6999495
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.8123529, upper bound: 14.6795541
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7922722, upper bound: 14.6996426
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7445044, upper bound: 14.7474006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7244316, upper bound: 14.7674930
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7448106, upper bound: 14.7470935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7247372, upper bound: 14.7671855
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7684591, upper bound: 14.7033546
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7684591, upper bound: 14.7234544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7888325, upper bound: 14.7030322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.79
Output dim: 2, lower bound: -14.7687493, upper bound: 14.7231289

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4316101, 54.4290466
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5547256, 36.5567436
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2135849, 33.2204514
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4806137, 32.4985428
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4733429, 43.4782639
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2199707, 38.2337570
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5468445, 44.5567551
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7676315, 41.7690887
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1610947, 47.1670532
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8358459, 35.8322449
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6566925, 50.6392136
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5182114, 49.5086136
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1193771, 44.1046600
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5924835, 49.5931320
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5724411, 48.5540848
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6395531, 40.6202965
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9719925, 29.9573975
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1670074, 29.1607246
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4640808, 40.4497910
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5272522, 38.5329208
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2854462, 35.2734108
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8610916, 32.8560410
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4251328, 34.4161797
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8097992, 43.8014221
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7802658, 37.7778549
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6678696, 36.6622925
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6941223, 37.6771622
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2487717, 57.2620544
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3524857, 42.3613205
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9592438, 43.9667358
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5858307, 45.5886536
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2529831, 55.2407608
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2175751, 53.2197800
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5912857, 45.5927124
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9080124, 36.9138145

Time for backsubstitution: 1.81 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7223840, upper bound: 14.7672332
time: 33.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7216116, upper bound: 14.7679996
time: 50.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4308624, 54.4297943
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5542831, 36.5571861
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2132568, 33.2207870
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4812393, 32.4979172
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4733887, 43.4782104
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2197571, 38.2339630
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5460815, 44.5575104
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7661972, 41.7705078
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1606064, 47.1675491
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8351212, 35.8329735
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6565247, 50.6393814
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5163193, 49.5104904
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1207047, 44.1033325
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5935059, 49.5921173
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5698166, 48.5566940
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6397057, 40.6201401
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9705582, 29.9588356
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1659164, 29.1618118
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4625702, 40.4513092
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5287018, 38.5314789
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2843704, 35.2744827
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8605957, 32.8565407
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4244919, 34.4168167
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8104401, 43.8007965
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7787857, 37.7793350
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6674500, 36.6627121
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6921082, 37.6791725
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2506943, 57.2601318
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3559952, 42.3578110
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9620056, 43.9639740
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5868073, 45.5876770
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2538834, 55.2398453
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2176056, 53.2197647
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5911942, 45.5927887
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9071350, 36.9146957

Time for backsubstitution: 1.81 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7022892, upper bound: 14.7873135
time: 42.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7015193, upper bound: 14.7880825
time: 33.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4316254, 54.4290314
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5555954, 36.5558739
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2150650, 33.2189713
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4822388, 32.4969254
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4743652, 43.4772491
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2230225, 38.2306900
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5517883, 44.5518036
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7705612, 41.7661438
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1631088, 47.1650467
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8358231, 35.8322754
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6538544, 50.6420364
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5180588, 49.5087662
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1142502, 44.1097870
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5910492, 49.5945816
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5723190, 48.5542068
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6388359, 40.6210098
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9706650, 29.9587250
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1674500, 29.1602745
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4633026, 40.4505692
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5272980, 38.5328674
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2847443, 35.2741089
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8609161, 32.8562164
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4247971, 34.4165115
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8092041, 43.8020325
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7808380, 37.7772827
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6678238, 36.6623383
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6939621, 37.6773186
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2488174, 57.2620163
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3521805, 42.3616333
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9580536, 43.9679260
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5854340, 45.5890579
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2467880, 55.2469482
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2193451, 53.2180328
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5914993, 45.5924988
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9115295, 36.9103012

Time for backsubstitution: 1.82 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7227099, upper bound: 14.7669439
time: 45.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7219380, upper bound: 14.7677093
time: 18.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4308777, 54.4297791
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5551529, 36.5563164
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2147369, 33.2193069
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4828720, 32.4962997
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4744110, 43.4772034
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2228088, 38.2309036
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5510254, 44.5525589
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7691574, 41.7675629
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1626205, 47.1655502
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8350906, 35.8330002
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6537018, 50.6422043
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5161667, 49.5106430
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1155624, 44.1084671
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5920563, 49.5935669
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5696945, 48.5568161
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6389961, 40.6208496
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9692307, 29.9601631
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1663666, 29.1613617
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4617767, 40.4520874
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5287323, 38.5314255
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2836685, 35.2751808
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8604126, 32.8567162
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4241638, 34.4171448
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8098297, 43.8013992
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7793579, 37.7787628
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6674042, 36.6627502
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6919556, 37.6793289
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2507401, 57.2600937
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3556900, 42.3581238
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9608002, 43.9651718
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5864105, 45.5880814
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2477036, 55.2460327
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2193451, 53.2180252
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5914078, 45.5925827
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9106522, 36.9111786

Time for backsubstitution: 1.80 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7026125, upper bound: 14.7870231
time: 32.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7018439, upper bound: 14.7877906
time: 36.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4337311, 54.4269257
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5589905, 36.5524712
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2196503, 33.2143936
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4944077, 32.4847488
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4754028, 43.4762115
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2295837, 38.2241211
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5518494, 44.5517426
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7694016, 41.7673111
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1669846, 47.1611710
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8308258, 35.8372688
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6413727, 50.6545258
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5151443, 49.5116730
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1080551, 44.1159744
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5939331, 49.5916901
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5653152, 48.5612030
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6231270, 40.6367188
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9644470, 29.9649429
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1576004, 29.1701241
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4551544, 40.4587097
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5267181, 38.5334473
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2776566, 35.2812004
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8569336, 32.8601990
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4179688, 34.4233398
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7957306, 43.8154984
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7757950, 37.7823257
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6637497, 36.6664047
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6838989, 37.6873856
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2603226, 57.2505112
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3557358, 42.3580704
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9684906, 43.9574814
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5858459, 45.5886459
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2514114, 55.2423248
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2178497, 53.2195206
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5933609, 45.5906296
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9101486, 36.9116821

Time for backsubstitution: 1.80 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7664328, upper bound: 14.7232135
time: 32.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7656678, upper bound: 14.7239852
time: 35.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4329834, 54.4276657
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5585480, 36.5529137
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2193222, 33.2147217
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4950409, 32.4841232
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4754486, 43.4761581
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2293854, 38.2243347
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5510864, 44.5524979
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7679672, 41.7687302
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1664963, 47.1616669
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8301010, 35.8379974
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6412201, 50.6546936
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5132675, 49.5135498
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1093826, 44.1146545
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5949402, 49.5906754
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5627060, 48.5638046
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6232872, 40.6365623
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9630127, 29.9663811
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1565170, 29.1712112
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4536591, 40.4602280
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5281525, 38.5320053
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2765808, 35.2822762
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8564377, 32.8606949
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4173355, 34.4239731
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7963562, 43.8148651
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7743149, 37.7838058
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6633377, 36.6668243
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6818848, 37.6893959
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2622452, 57.2485886
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3592606, 42.3545532
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9712524, 43.9547272
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5868225, 45.5876694
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2523270, 55.2414169
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2178497, 53.2195129
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5932846, 45.5907135
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9092636, 36.9125671

Time for backsubstitution: 1.81 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7463412, upper bound: 14.7432845
time: 35.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7455773, upper bound: 14.7440597
time: 44.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4337311, 54.4269104
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5598602, 36.5516014
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2211304, 33.2129135
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4960403, 32.4831314
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4764099, 43.4751892
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2326660, 38.2210617
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5567932, 44.5467911
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7723465, 41.7643661
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1689987, 47.1591568
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8308029, 35.8372955
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6385651, 50.6573486
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5149918, 49.5118256
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1029282, 44.1211090
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5924835, 49.5931396
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5651932, 48.5613251
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6224174, 40.6374321
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9631271, 29.9662704
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1580505, 29.1696777
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4543915, 40.4594879
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5267639, 38.5334015
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2769547, 35.2818985
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8567581, 32.8603745
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4176407, 34.4236679
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7951202, 43.8161011
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7763672, 37.7817535
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6637115, 36.6664429
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6837387, 37.6875420
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2603531, 57.2504730
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3554306, 42.3583832
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9673004, 43.9586716
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5854340, 45.5890503
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2452164, 55.2485123
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2195892, 53.2177811
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5935745, 45.5904160
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9136581, 36.9081726

Time for backsubstitution: 1.82 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7667402, upper bound: 14.7229092
time: 32.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7659739, upper bound: 14.7236806
time: 58.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4329987, 54.4276581
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5594177, 36.5520439
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2208023, 33.2132416
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4966660, 32.4824982
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4764709, 43.4751434
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2324524, 38.2212753
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5560303, 44.5475464
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7709274, 41.7657852
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1685104, 47.1596603
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8300705, 35.8380203
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6383820, 50.6575165
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5131149, 49.5137024
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1042557, 44.1197891
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5934906, 49.5921249
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5625839, 48.5639343
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6225700, 40.6372757
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9616852, 29.9677086
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1569672, 29.1707611
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4528656, 40.4610062
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5282135, 38.5319595
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2758789, 35.2829742
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8562622, 32.8608704
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4170074, 34.4243011
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.7957611, 43.8154755
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7748871, 37.7832336
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6632919, 36.6668625
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6817322, 37.6895523
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2622757, 57.2485504
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3589401, 42.3548737
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9700623, 43.9559174
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5864105, 45.5880737
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2461319, 55.2476044
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2195892, 53.2177658
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5934982, 45.5904999
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9127808, 36.9090500

Time for backsubstitution: 1.82 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7466483, upper bound: 14.7429779
time: 46.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458839, upper bound: 14.7437532
time: 39.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4270630, 54.4327850
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5500107, 36.5606194
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2104187, 33.2229767
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4784317, 32.5002136
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4724426, 43.4787598
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2176819, 38.2358856
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5471191, 44.5564423
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7590866, 41.7743683
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1570358, 47.1690216
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8329086, 35.8326950
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6572876, 50.6399460
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5103836, 49.5123672
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1276093, 44.0938110
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5941467, 49.5889587
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5606918, 48.5602264
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6402702, 40.6199303
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9664383, 29.9609985
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1669922, 29.1608315
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4569397, 40.4528732
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5299530, 38.5254135
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2818604, 35.2756310
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8595276, 32.8572121
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4244156, 34.4168625
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8177643, 43.7930908
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7789841, 37.7786179
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6668091, 36.6629257
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6864929, 37.6831551
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2516251, 57.2590942
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3613358, 42.3508148
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9640121, 43.9618454
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5924377, 45.5807343
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2585678, 55.2346649
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2203217, 53.2166595
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5914230, 45.5925446
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9071732, 36.9141350

Time for backsubstitution: 1.81 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6988905, upper bound: 14.7907528
time: 36.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6981169, upper bound: 14.7915200
time: 32.26 seconds

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

Time for backsubstitution: 1.73 seconds

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
time: 36.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6780333, upper bound: 14.8116010
time: 47.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4270782, 54.4327850
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5508804, 36.5597572
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2118988, 33.2214966
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4800491, 32.4985962
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4734650, 43.4777451
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2207336, 38.2328262
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5520630, 44.5514832
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7620316, 41.7714233
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1590500, 47.1670151
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8328857, 35.8327179
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6544647, 50.6427689
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5102310, 49.5125198
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1224670, 44.0989456
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5926971, 49.5904083
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5605698, 48.5603485
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6395607, 40.6206436
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9651108, 29.9623260
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1674423, 29.1603851
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4561615, 40.4536514
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5300140, 38.5253601
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2811661, 35.2763290
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8593521, 32.8573875
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4240799, 34.4171906
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8171539, 43.7936935
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7795563, 37.7780457
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6667709, 36.6629715
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6863403, 37.6833115
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2516708, 57.2590637
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3610153, 42.3511353
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9628220, 43.9630356
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5920410, 45.5811386
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2523727, 55.2408447
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2220612, 53.2149200
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5916367, 45.5923386
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9106827, 36.9106216

Time for backsubstitution: 1.72 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6991974, upper bound: 14.7904475
time: 41.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6984236, upper bound: 14.7912131
time: 34.68 seconds

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

Time for backsubstitution: 1.82 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6791082, upper bound: 14.8105261
time: 20.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6783370, upper bound: 14.8112942
time: 33.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4291840, 54.4306641
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5542831, 36.5563545
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2164841, 33.2169189
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4922256, 32.4864197
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4745178, 43.4766998
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2272949, 38.2262573
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5521240, 44.5514297
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7608566, 41.7725906
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1629257, 47.1631317
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8278885, 35.8377151
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6419830, 50.6552582
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5073318, 49.5154266
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1162872, 44.1051331
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5955811, 49.5875168
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5535812, 48.5673370
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6238518, 40.6363564
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9588928, 29.9685440
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1575928, 29.1702347
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4480133, 40.4617920
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5294342, 38.5259476
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2740707, 35.2834206
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8553696, 32.8613701
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4172516, 34.4240189
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8036957, 43.8071594
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7745209, 37.7830887
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6626968, 36.6670380
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6762695, 37.6933784
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2631760, 57.2475510
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3645859, 42.3475647
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9732590, 43.9525986
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5924377, 45.5807266
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2569809, 55.2362289
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2205658, 53.2164001
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5934982, 45.5904694
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9092941, 36.9120064

Time for backsubstitution: 1.81 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7429330, upper bound: 14.7467234
time: 33.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7421679, upper bound: 14.7474942
time: 37.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.9694290, 32.9071350, -23.9694290, 32.9071350, -54.4284515, 54.4314117
1: -7.7107468, 32.3472137, -7.7107468, 32.3472137, -36.5538406, 36.5567970
2: -4.8029060, 31.8072033, -4.8029060, 31.8072033, -33.2161484, 33.2172470
3: -8.9906178, 28.9257641, -8.9906178, 28.9257641, -32.4928513, 32.4857941
4: -10.1699419, 35.0947151, -10.1699419, 35.0947151, -43.4745636, 43.4766541
5: -11.1998892, 29.9470634, -11.1998892, 29.9470634, -38.2270813, 38.2264709
6: -38.7617340, 7.5140324, -38.7617340, 7.5140324, -44.5513611, 44.5521851
7: -15.3343515, 30.6479225, -15.3343515, 30.6479225, -41.7594376, 41.7740173
8: -15.5151711, 34.5915833, -15.5151711, 34.5915833, -47.1624374, 47.1636353
9: -10.3623190, 27.2287331, -10.3623190, 27.2287331, -35.8271637, 35.8384399
10: -28.4753380, 24.0453873, -28.4753380, 24.0453873, -50.6417999, 50.6554260
11: -35.8125153, 14.3663073, -35.8125153, 14.3663073, -49.5054550, 49.5173035
12: -49.4297676, 2.0979538, -49.4297676, 2.0979538, -44.1175995, 44.1038132
13: -28.8350906, 21.3669567, -28.8350906, 21.3669567, -49.5966034, 49.5865021
14: -71.0194244, -6.2968330, -71.0194244, -6.2968330, -64.7225952, 64.7225952
15: -17.4604645, 24.7060604, -17.4604645, 24.7060604, -42.1665268, 42.1665268
16: -27.5484943, 23.7199268, -27.5484943, 23.7199268, -48.5509720, 48.5699463
17: -71.1800537, -3.7373772, -71.1800537, -3.7373772, -67.4426727, 67.4426727
18: -34.7665024, 11.6985855, -34.7665024, 11.6985855, -40.6240044, 40.6362000
19: -25.7229214, 5.2951660, -25.7229214, 5.2951660, -29.9574585, 29.9699821
20: -26.4890232, 4.3974657, -26.4890232, 4.3974657, -29.1565094, 29.1713181
21: -31.3846607, 10.1913958, -31.3846607, 10.1913958, -40.4465027, 40.4633102
22: -33.6473465, 7.0107861, -33.6473465, 7.0107861, -38.5308685, 38.5244980
23: -26.9269905, 8.9186630, -26.9269905, 8.9186630, -35.2729950, 35.2844963
24: -23.2844276, 9.8805866, -23.2844276, 9.8805866, -32.8548737, 32.8618698
25: -29.2327003, 6.1404505, -29.2327003, 6.1404505, -34.4166183, 34.4246521
26: -42.9743996, 7.8795571, -42.9743996, 7.8795571, -43.8043213, 43.8065262
27: -26.6803741, 11.4951830, -26.6803741, 11.4951830, -37.7730408, 37.7845688
28: -29.6194363, 7.1861706, -29.6194363, 7.1861706, -36.6622849, 36.6674576
29: -32.6211929, 9.0433979, -32.6211929, 9.0433979, -41.6645889, 41.6645889
30: -37.6096840, 7.0090666, -37.6096840, 7.0090666, -44.6187515, 44.6187515
31: -31.4314880, 7.3121300, -31.4314880, 7.3121300, -37.6742630, 37.6953888
32: -33.7314987, 6.7933617, -33.7314987, 6.7933617, -40.5248604, 40.5248604
33: -44.0303955, 16.0379105, -44.0303955, 16.0379105, -57.2650986, 57.2456284
34: -50.7731895, -4.2341170, -50.7731895, -4.2341170, -42.3680954, 42.3440552
35: -41.0334129, 7.0879445, -41.0334129, 7.0879445, -43.9760208, 43.9498367
36: -44.5064888, 5.4577508, -44.5064888, 5.4577508, -45.5934143, 45.5797501
37: -59.4914551, 2.3993750, -59.4914551, 2.3993750, -55.2578964, 55.2353210
38: -50.9642067, 8.6528902, -50.9642067, 8.6528902, -59.6170959, 59.6170959
39: -52.2080841, 14.9309597, -52.2080841, 14.9309597, -67.1390457, 67.1390457
40: -47.9208145, 8.3381596, -47.9208145, 8.3381596, -53.2205963, 53.2163849
41: -31.9215584, 15.2862215, -31.9215584, 15.2862215, -45.5934372, 45.5905457
42: -27.2138786, 10.2254105, -27.2138786, 10.2254105, -36.9084167, 36.9128876

Time for backsubstitution: 1.82 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7228368, upper bound: 14.7668043
time: 41.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7220740, upper bound: 14.7675790
time: 40.21 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 83.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7223840, upper bound: 14.7672332
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7216116, upper bound: 14.7679996
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7022892, upper bound: 14.7873135
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7015193, upper bound: 14.7880825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7227099, upper bound: 14.7669439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7219380, upper bound: 14.7677093
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7026125, upper bound: 14.7870231
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7018439, upper bound: 14.7877906
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7664328, upper bound: 14.7232135
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7656678, upper bound: 14.7239852
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7463412, upper bound: 14.7432845
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7455773, upper bound: 14.7440597
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7667402, upper bound: 14.7229092
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7659739, upper bound: 14.7236806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7466483, upper bound: 14.7429779
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7458839, upper bound: 14.7437532
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.6988905, upper bound: 14.7907528
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.6981169, upper bound: 14.7915200
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.6788040, upper bound: 14.8108316
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.6780333, upper bound: 14.8116010
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.6991974, upper bound: 14.7904475
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.6984236, upper bound: 14.7912131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.6791082, upper bound: 14.8105261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.6783370, upper bound: 14.8112942
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7429330, upper bound: 14.7467234
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7421679, upper bound: 14.7474942
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7228368, upper bound: 14.7668043
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 83.72
Output dim: 2, lower bound: -14.7220740, upper bound: 14.7675790
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7439767, upper bound: 14.7479138
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7238778, upper bound: 14.7679981
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7679980, upper bound: 14.7238778
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7479137, upper bound: 14.7439768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7683251, upper bound: 14.7235871
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7482387, upper bound: 14.7436841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7919652, upper bound: 14.6798602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7919653, upper bound: 14.6999495
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.8123529, upper bound: 14.6795541
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7922722, upper bound: 14.6996426
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7445044, upper bound: 14.7474006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7244316, upper bound: 14.7674930
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7448106, upper bound: 14.7470935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7247372, upper bound: 14.7671855
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7684591, upper bound: 14.7033546
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7684591, upper bound: 14.7234544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7888325, upper bound: 14.7030322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 83.72
Output dim: 2, lower bound: -14.7687493, upper bound: 14.7231289

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 50.16 + 3586.34 = 3636.51 seconds
