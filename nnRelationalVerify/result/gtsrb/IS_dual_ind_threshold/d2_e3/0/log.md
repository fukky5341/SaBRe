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
execution time: IAR + RelationalAnalysis = 2.80 + 47.04 = 49.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -14.8236368, upper bound: 14.8236369

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7831672, upper bound: 14.8227297
time: 42.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8227295, upper bound: 14.8227297
time: 41.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 83.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 83.94
Output dim: 2, lower bound: -14.7831672, upper bound: 14.8227297
IS_A2, status: Status.UNKNOWN, split count: 1, time: 83.94
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

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
time: 39.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455
time: 19.59 seconds

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

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
time: 19.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455
time: 40.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 62.65 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 62.65
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 62.65
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 62.65
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 62.65
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -23.8177013, 32.8101044, -23.9083443, 32.8879089, -54.2716370, 54.2548370
1: -7.6333418, 32.3247108, -7.6789436, 32.3440895, -36.4774323, 36.4860077
2: -4.6688461, 31.7234459, -4.7409310, 31.8031101, -33.1046143, 33.0741730
3: -8.8590059, 28.8418007, -8.9294968, 28.9131241, -32.4015656, 32.3940544
4: -10.0128727, 35.0116959, -10.0950279, 35.0854683, -43.3382721, 43.3421860
5: -11.0313568, 29.8295517, -11.1212673, 29.9287872, -38.0783234, 38.0548248
6: -38.6645508, 7.4382792, -38.7291679, 7.4780035, -44.3220901, 44.4097366
7: -15.2223110, 30.6138916, -15.2916164, 30.6377048, -41.6822052, 41.6940918
8: -15.3571224, 34.4789886, -15.4382401, 34.5776787, -47.0213928, 46.9828491
9: -10.2696791, 27.0731945, -10.3508596, 27.1583576, -35.7069855, 35.7016754
10: -28.2383347, 23.7223549, -28.4552288, 23.8902054, -50.3182755, 50.3389206
11: -35.5332413, 14.1156454, -35.7802086, 14.2414227, -49.1236725, 49.2524109
12: -49.2245789, 1.7549562, -49.4142761, 1.9343047, -43.8075180, 43.8535919
13: -28.7948818, 21.2192383, -28.8224068, 21.3104515, -49.5271912, 49.4526901
14: -70.8596649, -6.5276146, -70.9999008, -6.4058208, -64.4538422, 64.4722900
15: -17.3281364, 24.5761299, -17.3987045, 24.6812801, -42.0094147, 41.9748344
16: -27.3626747, 23.5477905, -27.5174980, 23.6340294, -48.3501053, 48.3700485
17: -71.1006470, -3.9659004, -71.1615677, -3.8446007, -67.2560425, 67.1956635
18: -34.6748276, 11.6754570, -34.7447243, 11.6881580, -40.5022850, 40.5595245
19: -25.6248779, 5.2308908, -25.7009296, 5.2612734, -29.8632965, 29.8951263
20: -26.3760929, 4.2926950, -26.4702148, 4.3461385, -28.9960861, 29.0308189
21: -31.2165127, 10.0487394, -31.3613243, 10.1185780, -40.2115021, 40.3106689
22: -33.5701408, 6.9341712, -33.6168213, 6.9781919, -38.4134979, 38.4454727
23: -26.8236217, 8.8422222, -26.9121208, 8.8823624, -35.1975174, 35.2156677
24: -23.1826973, 9.8588467, -23.2450409, 9.8736801, -32.7443390, 32.7761536
25: -29.1614876, 6.0594473, -29.2123680, 6.1035247, -34.3512726, 34.3267250
26: -42.8395920, 7.6982255, -42.9475975, 7.7956986, -43.5716934, 43.6229248
27: -26.5719299, 11.4782562, -26.6374359, 11.4881344, -37.6916962, 37.7573051
28: -29.5719452, 7.1464739, -29.6025505, 7.1710525, -36.5971756, 36.5840530
29: -32.5674820, 8.9558697, -32.6080437, 9.0019608, -41.5694427, 41.5639114
30: -37.5125389, 6.8950462, -37.5880280, 6.9571342, -44.4696732, 44.4830742
31: -31.3037453, 7.2667732, -31.4060097, 7.2884336, -37.5703049, 37.5878754
32: -33.6125793, 6.6667624, -33.7145691, 6.7331514, -40.3457298, 40.3813324
33: -43.8413963, 15.9198933, -43.9421883, 16.0154305, -57.0658875, 57.0815582
34: -50.6624031, -4.3120794, -50.7218933, -4.2464752, -42.2542419, 42.2828751
35: -40.8815308, 6.9855962, -40.9614410, 7.0736732, -43.8322220, 43.8385162
36: -44.4452286, 5.3975372, -44.4803581, 5.4362044, -45.5112000, 45.5307541
37: -59.3934708, 2.3479176, -59.4541092, 2.3793426, -55.1153259, 55.1525803
38: -50.8613586, 8.5674448, -50.9221191, 8.6161909, -59.4775505, 59.4895630
39: -52.1121216, 14.8362160, -52.1709023, 14.8967543, -67.0088730, 67.0071182
40: -47.8043747, 8.2948952, -47.8753929, 8.3255510, -53.1139603, 53.1380463
41: -31.7976608, 15.2262154, -31.8880901, 15.2580700, -45.4287949, 45.5031967
42: -27.0813942, 10.0602674, -27.1930885, 10.1448975, -36.6113663, 36.6768341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=179, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
time: 72.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
time: 37.47 seconds

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

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
time: 40.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8189531, upper bound: 14.8189535
time: 77.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 119.74 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 119.74
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 119.74
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 119.74
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 119.74
Output dim: 2, lower bound: -14.8189531, upper bound: 14.8189535

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -23.7669926, 32.8026619, -23.8966351, 32.8862457, -54.2186661, 54.2353363
1: -7.6093254, 32.3227196, -7.6735058, 32.3436394, -36.4536667, 36.4775963
2: -4.6226931, 31.7213650, -4.7303529, 31.8026581, -33.0592117, 33.0610428
3: -8.8022289, 28.8371201, -8.9159451, 28.9121094, -32.3452606, 32.3766403
4: -9.9774189, 35.0078354, -10.0869541, 35.0845642, -43.3038406, 43.3302841
5: -10.9666491, 29.8226376, -11.1060772, 29.9272118, -38.0126495, 38.0329437
6: -38.6504822, 7.4307241, -38.7259979, 7.4762535, -44.3011169, 44.3929367
7: -15.1779766, 30.6093998, -15.2814445, 30.6366978, -41.6359482, 41.6790771
8: -15.3117828, 34.4739799, -15.4278965, 34.5765228, -46.9768753, 46.9673996
9: -10.2645359, 27.0498428, -10.3496628, 27.1530533, -35.6966629, 35.6793327
10: -28.2324371, 23.6489754, -28.4538345, 23.8735466, -50.2952347, 50.2644272
11: -35.5183716, 14.0622873, -35.7768135, 14.2294064, -49.0967255, 49.1947708
12: -49.2188187, 1.6442060, -49.4129753, 1.9090385, -43.7762985, 43.7412567
13: -28.7796898, 21.1974869, -28.8189507, 21.3055000, -49.5071564, 49.4274139
14: -70.8500595, -6.5967178, -70.9976654, -6.4215164, -64.4285431, 64.4009476
15: -17.3101826, 24.5638752, -17.3946552, 24.6785583, -41.9887390, 41.9585304
16: -27.3490028, 23.5268211, -27.5144768, 23.6292400, -48.3304749, 48.3522873
17: -71.0910568, -4.0345192, -71.1593323, -3.8602486, -67.2308044, 67.1248169
18: -34.6669960, 11.6452866, -34.7429237, 11.6813116, -40.4862976, 40.5274506
19: -25.6146393, 5.2079716, -25.6986198, 5.2560196, -29.8479462, 29.8732986
20: -26.3668213, 4.2652831, -26.4681549, 4.3399138, -28.9800110, 29.0045891
21: -31.2041225, 10.0030289, -31.3584843, 10.1082144, -40.1887436, 40.2632751
22: -33.5611115, 6.8992028, -33.6147690, 6.9703059, -38.3964005, 38.4075012
23: -26.8160133, 8.8251934, -26.9103966, 8.8785057, -35.1849060, 35.1997147
24: -23.1708813, 9.8484211, -23.2423248, 9.8713379, -32.7298889, 32.7646675
25: -29.1529007, 6.0345860, -29.2104073, 6.0977893, -34.3375473, 34.3042603
26: -42.8280220, 7.6072264, -42.9449921, 7.7750630, -43.5392456, 43.5298233
27: -26.5567780, 11.4622040, -26.6339531, 11.4844742, -37.6721878, 37.7386322
28: -29.5629501, 7.1372151, -29.6004848, 7.1689434, -36.5859299, 36.5744476
29: -32.5598831, 8.9210691, -32.6063156, 8.9940739, -41.5539551, 41.5273857
30: -37.5019302, 6.8705940, -37.5856476, 6.9513092, -44.4532394, 44.4562416
31: -31.2914391, 7.2515678, -31.4032211, 7.2849941, -37.5537758, 37.5741234
32: -33.6063385, 6.6453524, -33.7131844, 6.7282248, -40.3345642, 40.3585358
33: -43.7957993, 15.9102497, -43.9319496, 16.0132713, -57.0182800, 57.0614090
34: -50.6368027, -4.3180809, -50.7161064, -4.2478738, -42.2271118, 42.2709351
35: -40.8469200, 6.9806347, -40.9535904, 7.0725493, -43.7950134, 43.8250504
36: -44.4377861, 5.3832278, -44.4786835, 5.4329062, -45.4983368, 45.5129700
37: -59.3796158, 2.3326268, -59.4510002, 2.3758316, -55.0932617, 55.1312408
38: -50.8464813, 8.5507755, -50.9187813, 8.6123381, -59.4588203, 59.4695587
39: -52.0979462, 14.8208265, -52.1676826, 14.8931580, -66.9911041, 66.9885101
40: -47.7825012, 8.2895107, -47.8703842, 8.3243256, -53.0918961, 53.1281509
41: -31.7838249, 15.2158155, -31.8849945, 15.2556620, -45.4106903, 45.4861984
42: -27.0720615, 10.0426826, -27.1909981, 10.1403675, -36.5938950, 36.6420555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
time: 33.96 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7415511, upper bound: 14.8146118
time: 30.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -23.8521843, 32.9531059, -23.8998947, 32.8866653, -54.3062286, 54.3970108
1: -7.6428223, 32.3827896, -7.6730280, 32.3433647, -36.4928436, 36.5330124
2: -4.6835995, 31.8502617, -4.7362261, 31.8021545, -33.1185455, 33.2048531
3: -8.8735180, 29.0136604, -8.9234972, 28.9114532, -32.4087524, 32.5537949
4: -10.0240583, 35.1153946, -10.0901251, 35.0839767, -43.3509521, 43.4439545
5: -11.0468864, 30.0121708, -11.1151457, 29.9269981, -38.0881424, 38.2330093
6: -38.7042236, 7.4491806, -38.7257690, 7.4729137, -44.3883820, 44.4107132
7: -15.2509117, 30.7087803, -15.2835636, 30.6353340, -41.7048950, 41.7874756
8: -15.3604450, 34.5904236, -15.4301052, 34.5762253, -47.0279388, 47.1014099
9: -10.3270674, 27.0874062, -10.3487225, 27.1515656, -35.7516632, 35.7222710
10: -28.4464607, 23.7343121, -28.4538841, 23.8805389, -50.5168228, 50.3478699
11: -35.7413025, 14.1150599, -35.7765694, 14.2343521, -49.3277206, 49.2451935
12: -49.4987030, 1.7658911, -49.4115105, 1.9206252, -44.0748978, 43.8569870
13: -28.8023472, 21.3003254, -28.8150597, 21.3068314, -49.5305786, 49.5207138
14: -71.0530548, -6.5266953, -70.9964752, -6.4144459, -64.6386108, 64.4697800
15: -17.3276920, 24.6473579, -17.3849258, 24.6787529, -42.0064468, 42.0322838
16: -27.4442959, 23.5374222, -27.5137024, 23.6256351, -48.3817749, 48.3620453
17: -71.2382660, -3.9546680, -71.1582794, -3.8511906, -67.3870773, 67.2036133
18: -34.8298340, 11.6764889, -34.7426643, 11.6812916, -40.6591492, 40.5707703
19: -25.7185192, 5.2272806, -25.6989098, 5.2580891, -29.9348526, 29.8886032
20: -26.4766636, 4.2961893, -26.4686871, 4.3421068, -29.0967712, 29.0371666
21: -31.3783035, 10.0458775, -31.3584538, 10.1129351, -40.3778305, 40.3059998
22: -33.6378708, 6.9429760, -33.6134071, 6.9677033, -38.4793472, 38.4515915
23: -26.9233494, 8.8503704, -26.9103928, 8.8799324, -35.2755737, 35.2293549
24: -23.2352428, 9.8638287, -23.2421570, 9.8689861, -32.7805519, 32.7812119
25: -29.2197533, 6.0668054, -29.2097969, 6.0996490, -34.3874817, 34.3380432
26: -43.1041145, 7.7142849, -42.9437637, 7.7842088, -43.8278198, 43.6335068
27: -26.6342258, 11.4793425, -26.6346283, 11.4834614, -37.7457504, 37.7569771
28: -29.6283989, 7.1627240, -29.6007729, 7.1680918, -36.6391678, 36.6144638
29: -32.6346130, 8.9576969, -32.6050453, 8.9958296, -41.6304436, 41.5627441
30: -37.6130753, 6.9097223, -37.5854568, 6.9528894, -44.5659637, 44.4951782
31: -31.4241123, 7.2671928, -31.4038391, 7.2860141, -37.6397400, 37.6021080
32: -33.6860390, 6.6740026, -33.7127609, 6.7272224, -40.4132614, 40.3867645
33: -43.8493843, 16.0234623, -43.9358749, 16.0130005, -57.0716248, 57.1922989
34: -50.6684914, -4.2364578, -50.7181282, -4.2480392, -42.2556686, 42.3514328
35: -40.8835869, 7.0813065, -40.9563141, 7.0727491, -43.8310013, 43.9246140
36: -44.4544144, 5.4209433, -44.4779968, 5.4309449, -45.5170822, 45.5555267
37: -59.4538956, 2.3641095, -59.4505081, 2.3718228, -55.1617432, 55.1814499
38: -50.8916817, 8.5897112, -50.9181213, 8.6090412, -59.5007248, 59.5078316
39: -52.1484833, 14.8712988, -52.1679611, 14.8893089, -67.0377960, 67.0392609
40: -47.8347130, 8.3285494, -47.8709450, 8.3226051, -53.1454391, 53.1703262
41: -31.8363914, 15.2370930, -31.8850269, 15.2524557, -45.4777679, 45.5048676
42: -27.1478577, 10.0665112, -27.1907539, 10.1353970, -36.7332306, 36.6796341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=179, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7436501, upper bound: 14.8146118
time: 32.47 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7727093, upper bound: 14.8146118
time: 41.07 seconds

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

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
time: 35.08 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
time: 46.10 seconds

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

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7855901, upper bound: 14.8146118
time: 39.52 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8146113, upper bound: 14.8146118
time: 36.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 77.74 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 77.74
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 77.74
Output dim: 2, lower bound: -14.7415511, upper bound: 14.8146118
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 77.74
Output dim: 2, lower bound: -14.7436501, upper bound: 14.8146118
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 77.74
Output dim: 2, lower bound: -14.7727093, upper bound: 14.8146118
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 77.74
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 77.74
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 77.74
Output dim: 2, lower bound: -14.7855901, upper bound: 14.8146118
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 77.74
Output dim: 2, lower bound: -14.8146113, upper bound: 14.8146118

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -23.7623119, 32.7893410, -23.8858871, 32.8552589, -54.1829453, 54.2107925
1: -7.6070089, 32.3096352, -7.6681376, 32.3135300, -36.4217453, 36.4595566
2: -4.6195107, 31.7015800, -4.7230015, 31.7576199, -33.0107269, 33.0341759
3: -8.7989359, 28.8318348, -8.9083595, 28.8998680, -32.3271713, 32.3619843
4: -9.9745111, 34.9884224, -10.0802860, 35.0396423, -43.2555084, 43.3038864
5: -10.9635859, 29.8143024, -11.0990343, 29.9078789, -37.9863968, 38.0152359
6: -38.6177979, 7.4279308, -38.6515427, 7.4698572, -44.2613602, 44.3150406
7: -15.1752815, 30.5950775, -15.2752590, 30.6040897, -41.5998077, 41.6586456
8: -15.3083696, 34.4553986, -15.4200735, 34.5334282, -46.9302521, 46.9402771
9: -10.2588720, 27.0458183, -10.3366480, 27.1437531, -35.6800766, 35.6627617
10: -28.2294083, 23.6277561, -28.4468746, 23.8245773, -50.2307892, 50.2294769
11: -35.5144501, 14.0483913, -35.7678909, 14.1978235, -49.0608521, 49.1711502
12: -49.2055550, 1.6398625, -49.3826332, 1.8989787, -43.7480316, 43.6996078
13: -28.7744293, 21.1929817, -28.8069439, 21.2949696, -49.4871750, 49.4084396
14: -70.8449402, -6.6288528, -70.9859848, -6.4959564, -64.3489838, 64.3571320
15: -17.3074913, 24.5583096, -17.3884487, 24.6658173, -41.9733086, 41.9467583
16: -27.3456764, 23.5202255, -27.5067253, 23.6143665, -48.3067017, 48.3353577
17: -71.0831146, -4.0843754, -71.1411896, -3.9757481, -67.1073685, 67.0568161
18: -34.6628723, 11.6373940, -34.7335205, 11.6639500, -40.4657822, 40.5104446
19: -25.6107788, 5.2014318, -25.6898174, 5.2410240, -29.8303223, 29.8578568
20: -26.3637180, 4.2627025, -26.4610405, 4.3340073, -28.9705887, 28.9947662
21: -31.2005234, 9.9962006, -31.3501816, 10.0925140, -40.1713867, 40.2484055
22: -33.5572433, 6.8938112, -33.6058350, 6.9579134, -38.3709946, 38.3876572
23: -26.8126316, 8.8188314, -26.9025784, 8.8639154, -35.1680145, 35.1853485
24: -23.1660595, 9.8372469, -23.2312031, 9.8456306, -32.6994705, 32.7420654
25: -29.1496696, 6.0233245, -29.2030296, 6.0720949, -34.3077698, 34.2848015
26: -42.8231583, 7.6045756, -42.9339256, 7.7689633, -43.5265350, 43.5136414
27: -26.5412483, 11.4601402, -26.5979729, 11.4797859, -37.6492920, 37.6968689
28: -29.5567093, 7.1336360, -29.5862751, 7.1607347, -36.5689087, 36.5539398
29: -32.5543251, 8.9164143, -32.5936584, 8.9833593, -41.5376854, 41.5100708
30: -37.4997101, 6.8642464, -37.5805931, 6.9368401, -44.4365501, 44.4448395
31: -31.2863007, 7.2374473, -31.3914375, 7.2525973, -37.5189400, 37.5489502
32: -33.5723724, 6.6422405, -33.6345444, 6.7211065, -40.2934799, 40.2767868
33: -43.7886734, 15.9079456, -43.9155960, 16.0079498, -57.0006256, 57.0358734
34: -50.6125984, -4.3204756, -50.6601257, -4.2533159, -42.1974869, 42.2132263
35: -40.8329964, 6.9795365, -40.9218178, 7.0699987, -43.7782059, 43.7931595
36: -44.4027710, 5.3812938, -44.3977509, 5.4283996, -45.4586563, 45.4296951
37: -59.3600616, 2.3281960, -59.4072647, 2.3657165, -55.0612030, 55.0766602
38: -50.8162498, 8.5480871, -50.8492165, 8.6061029, -59.4223518, 59.3973045
39: -52.0838661, 14.8181477, -52.1354790, 14.8869381, -66.9708023, 66.9536285
40: -47.7543907, 8.2863836, -47.8057137, 8.3169470, -53.0550003, 53.0582733
41: -31.7518883, 15.2128420, -31.8111839, 15.2487688, -45.3717194, 45.4090958
42: -27.0496349, 10.0395155, -27.1392956, 10.1331100, -36.5639725, 36.5869293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 40.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 14.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -23.7644920, 32.7965431, -23.9595127, 32.8853416, -54.2163620, 54.2914734
1: -7.6079559, 32.3176880, -7.7193756, 32.3412247, -36.4467468, 36.5203972
2: -4.6213756, 31.7174530, -4.8156600, 31.8063145, -33.0588455, 33.1426315
3: -8.8008280, 28.8341656, -8.9563999, 28.9162712, -32.3429871, 32.4153671
4: -9.9761353, 35.0037727, -10.1723509, 35.0896225, -43.3060303, 43.4101257
5: -10.9654779, 29.8184891, -11.1451292, 29.9303150, -38.0087128, 38.0746384
6: -38.6454277, 7.4292984, -38.7393913, 7.6039772, -44.4225311, 44.3993073
7: -15.1768246, 30.6062775, -15.3355703, 30.6389866, -41.6351700, 41.7305832
8: -15.3103828, 34.4706039, -15.5021534, 34.5833702, -46.9789505, 47.0378952
9: -10.2562771, 27.0477581, -10.3654671, 27.1777706, -35.7141418, 35.6982613
10: -28.2314911, 23.6391640, -28.5260677, 23.8763599, -50.2843475, 50.3608551
11: -35.5158539, 14.0578842, -35.8155022, 14.2375612, -49.1029739, 49.2272186
12: -49.2154922, 1.6416702, -49.4210739, 1.9429069, -43.8087692, 43.7435837
13: -28.7774048, 21.1934433, -28.8484116, 21.3225079, -49.5147781, 49.4602890
14: -70.8469925, -6.6035328, -71.1271362, -6.4205837, -64.4264069, 64.5236053
15: -17.3089218, 24.5604744, -17.4222031, 24.6824989, -41.9914207, 41.9826775
16: -27.3468399, 23.5233192, -27.5643463, 23.6409836, -48.3354340, 48.3972168
17: -71.0880203, -4.0442657, -71.3320999, -3.8626976, -67.2253265, 67.2878342
18: -34.6645279, 11.6359072, -34.7764587, 11.6876736, -40.4962997, 40.5494347
19: -25.6123734, 5.2045655, -25.7233734, 5.2641163, -29.8535004, 29.8886757
20: -26.3638935, 4.2639499, -26.4715290, 4.3571038, -28.9923630, 29.0098724
21: -31.2013435, 9.9989414, -31.3805542, 10.1159134, -40.1953278, 40.2761765
22: -33.5591965, 6.8966398, -33.6409569, 6.9830422, -38.3969116, 38.4453354
23: -26.8138790, 8.8211384, -26.9291458, 8.8881979, -35.1944199, 35.2133636
24: -23.1672325, 9.8429451, -23.2880383, 9.8817711, -32.7355461, 32.8027344
25: -29.1500282, 6.0281014, -29.2520103, 6.1068425, -34.3446350, 34.3446960
26: -42.8249054, 7.6038866, -42.9597473, 7.7820683, -43.5472717, 43.5447998
27: -26.5500221, 11.4610443, -26.6440487, 11.5437565, -37.7424545, 37.7472420
28: -29.5574474, 7.1357331, -29.6048145, 7.2032218, -36.6173782, 36.5747223
29: -32.5553970, 8.9193287, -32.6248589, 9.0081272, -41.5635223, 41.5441895
30: -37.4996338, 6.8670073, -37.5951195, 6.9654560, -44.4650879, 44.4621277
31: -31.2886562, 7.2473598, -31.4300900, 7.2930112, -37.5556297, 37.5885391
32: -33.5996361, 6.6430244, -33.7248039, 6.8579178, -40.4575539, 40.3678284
33: -43.7912979, 15.9093475, -43.9397125, 16.0517654, -57.0612946, 57.0687103
34: -50.6295624, -4.3196754, -50.7220078, -4.1626968, -42.3136444, 42.2758484
35: -40.8415909, 6.9802766, -40.9592514, 7.1176510, -43.8418732, 43.8290939
36: -44.4305344, 5.3820791, -44.4855347, 5.5513158, -45.6095200, 45.5173645
37: -59.3710442, 2.3298440, -59.4542122, 2.4488144, -55.1658936, 55.1261292
38: -50.8370781, 8.5494165, -50.9301910, 8.7109747, -59.5480537, 59.4796066
39: -52.0915604, 14.8190231, -52.1798477, 14.9351101, -67.0266724, 66.9988708
40: -47.7764244, 8.2876635, -47.8778763, 8.4354010, -53.1954651, 53.1298676
41: -31.7770157, 15.2139101, -31.8973274, 15.3820343, -45.5311432, 45.4943771
42: -27.0673218, 10.0409164, -27.2018280, 10.2334347, -36.6818085, 36.6476135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6951250, upper bound: 14.8130237
time: 36.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 33.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -23.8475170, 32.9397278, -23.8891640, 32.8557129, -54.2704620, 54.3725204
1: -7.6405230, 32.3696899, -7.6677227, 32.3132668, -36.4609146, 36.5149612
2: -4.6803980, 31.8304138, -4.7288933, 31.7571354, -33.0700684, 33.1780014
3: -8.8702164, 29.0083714, -8.9159479, 28.8992500, -32.3906403, 32.5391808
4: -10.0211592, 35.0959549, -10.0834208, 35.0390892, -43.3026505, 43.4175110
5: -11.0438175, 30.0037937, -11.1080933, 29.9075890, -38.0618896, 38.2153320
6: -38.6715813, 7.4464149, -38.6513901, 7.4665451, -44.3486328, 44.3328171
7: -15.2482233, 30.6944695, -15.2774048, 30.6027679, -41.6687469, 41.7670135
8: -15.3570518, 34.5718231, -15.4222794, 34.5331573, -46.9813156, 47.0742874
9: -10.3213882, 27.0834141, -10.3356743, 27.1423187, -35.7350311, 35.7057228
10: -28.4433994, 23.7130489, -28.4469109, 23.8315392, -50.4523773, 50.3128967
11: -35.7373505, 14.1011391, -35.7675858, 14.2027454, -49.2918625, 49.2215729
12: -49.4854774, 1.7615199, -49.3811684, 1.9106312, -44.0466614, 43.8153839
13: -28.7970810, 21.2957993, -28.8029556, 21.2963638, -49.5105667, 49.5017700
14: -71.0479736, -6.5588760, -70.9847565, -6.4888515, -64.5591202, 64.4258804
15: -17.3249741, 24.6418133, -17.3787422, 24.6659584, -41.9909325, 42.0205536
16: -27.4409103, 23.5308266, -27.5059357, 23.6107140, -48.3579254, 48.3451309
17: -71.2303619, -4.0045357, -71.1401596, -3.9666824, -67.2636795, 67.1356201
18: -34.8257408, 11.6686001, -34.7332497, 11.6639271, -40.6386566, 40.5538177
19: -25.7146988, 5.2207441, -25.6901150, 5.2430840, -29.9172363, 29.8731689
20: -26.4735889, 4.2936459, -26.4615555, 4.3361983, -29.0872993, 29.0273895
21: -31.3746815, 10.0390215, -31.3501015, 10.0972252, -40.3604889, 40.2910919
22: -33.6340256, 6.9375868, -33.6045036, 6.9553657, -38.4539185, 38.4317474
23: -26.9199543, 8.8439865, -26.9025803, 8.8653183, -35.2586823, 35.2150269
24: -23.2304268, 9.8526649, -23.2310295, 9.8432636, -32.7501564, 32.7586670
25: -29.2165546, 6.0555201, -29.2024078, 6.0739832, -34.3577042, 34.3185577
26: -43.0993080, 7.7115679, -42.9327278, 7.7780514, -43.8151093, 43.6174088
27: -26.6186829, 11.4772902, -26.5986309, 11.4787407, -37.7228012, 37.7151871
28: -29.6221485, 7.1591611, -29.5865841, 7.1598706, -36.6222305, 36.5939331
29: -32.6291161, 8.9530735, -32.5923615, 8.9851208, -41.6142349, 41.5454330
30: -37.6108246, 6.9033880, -37.5803680, 6.9383430, -44.5491676, 44.4837570
31: -31.4189854, 7.2531037, -31.3920403, 7.2536364, -37.6048813, 37.5768967
32: -33.6520920, 6.6708984, -33.6341057, 6.7201004, -40.3721924, 40.3050041
33: -43.8422661, 16.0211678, -43.9195404, 16.0077019, -57.0539551, 57.1667023
34: -50.6442566, -4.2388177, -50.6622162, -4.2535038, -42.2260132, 42.2937698
35: -40.8696365, 7.0802169, -40.9245911, 7.0701842, -43.8141937, 43.8927155
36: -44.4194489, 5.4189901, -44.3970833, 5.4264002, -45.4774246, 45.4722900
37: -59.4343719, 2.3596501, -59.4067993, 2.3616323, -55.1296997, 55.1268311
38: -50.8615189, 8.5869646, -50.8485565, 8.6026783, -59.4641953, 59.4355202
39: -52.1343384, 14.8686094, -52.1357231, 14.8831396, -67.0174789, 67.0043335
40: -47.8066444, 8.3253250, -47.8062744, 8.3152752, -53.1085587, 53.1003799
41: -31.8044796, 15.2341080, -31.8111877, 15.2455864, -45.4388123, 45.4278412
42: -27.1255093, 10.0633545, -27.1390553, 10.1281109, -36.7033234, 36.6245193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6972172, upper bound: 14.8130237
time: 39.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7420616, upper bound: 14.8130237
time: 31.27 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.8496780, 32.9469795, -23.9627304, 32.8857498, -54.3038712, 54.4531784
1: -7.6414776, 32.3777275, -7.7189102, 32.3409615, -36.4858780, 36.5757980
2: -4.6822767, 31.8462944, -4.8215246, 31.8058205, -33.1181870, 33.2864380
3: -8.8721123, 29.0107231, -8.9639883, 28.9156170, -32.4064865, 32.5925865
4: -10.0227957, 35.1113205, -10.1755009, 35.0890388, -43.3531723, 43.5237656
5: -11.0456934, 30.0080185, -11.1542110, 29.9300270, -38.0842056, 38.2746887
6: -38.6991768, 7.4477892, -38.7392426, 7.6006279, -44.5098038, 44.4170761
7: -15.2497587, 30.7056541, -15.3377171, 30.6376877, -41.7041168, 41.8389587
8: -15.3590088, 34.5870438, -15.5043344, 34.5830841, -47.0300446, 47.1718826
9: -10.3188210, 27.0853214, -10.3645000, 27.1763687, -35.7691193, 35.7412071
10: -28.4455090, 23.7245026, -28.5261116, 23.8833752, -50.5059052, 50.4442749
11: -35.7387161, 14.1105986, -35.8152237, 14.2424698, -49.3340073, 49.2776947
12: -49.4953842, 1.7634068, -49.4196320, 1.9545279, -44.1073456, 43.8593521
13: -28.8000679, 21.2962189, -28.8444500, 21.3238239, -49.5382080, 49.5535965
14: -71.0500488, -6.5335274, -71.1259232, -6.4135113, -64.6365356, 64.5923920
15: -17.3263969, 24.6439400, -17.4124870, 24.6826591, -42.0090561, 42.0564270
16: -27.4420757, 23.5339508, -27.5635490, 23.6371803, -48.3867340, 48.4069901
17: -71.2352753, -3.9644585, -71.3310776, -3.8535900, -67.3816833, 67.3666229
18: -34.8273392, 11.6670818, -34.7762032, 11.6876774, -40.6691818, 40.5927963
19: -25.7162552, 5.2238832, -25.7236671, 5.2661543, -29.9404831, 29.9040413
20: -26.4737434, 4.2948604, -26.4720554, 4.3593302, -29.1090851, 29.0424652
21: -31.3755207, 10.0417500, -31.3805008, 10.1206303, -40.3843994, 40.3189240
22: -33.6359711, 6.9403863, -33.6396332, 6.9804997, -38.4798431, 38.4894104
23: -26.9212093, 8.8462925, -26.9291210, 8.8896008, -35.2850952, 35.2430344
24: -23.2315826, 9.8583355, -23.2878723, 9.8794355, -32.7862396, 32.8192902
25: -29.2169018, 6.0603099, -29.2513714, 6.1087356, -34.3946228, 34.3784790
26: -43.1009750, 7.7109394, -42.9585190, 7.7912169, -43.8358154, 43.6486053
27: -26.6274376, 11.4781742, -26.6447201, 11.5427275, -37.8159637, 37.7656021
28: -29.6229115, 7.1612730, -29.6051006, 7.2023511, -36.6707306, 36.6147842
29: -32.6301651, 8.9559727, -32.6235733, 9.0099106, -41.6400757, 41.5795441
30: -37.6107483, 6.9061317, -37.5948944, 6.9669323, -44.5776825, 44.5010262
31: -31.4213753, 7.2629933, -31.4307079, 7.2940612, -37.6415863, 37.6165009
32: -33.6792946, 6.6716967, -33.7244110, 6.8569746, -40.5362701, 40.3961067
33: -43.8448601, 16.0224895, -43.9437141, 16.0515175, -57.1145782, 57.1996155
34: -50.6612778, -4.2380009, -50.7240639, -4.1628728, -42.3421860, 42.3563843
35: -40.8782806, 7.0809407, -40.9619904, 7.1178689, -43.8778229, 43.9286575
36: -44.4471741, 5.4197388, -44.4849052, 5.5493813, -45.6283035, 45.5599976
37: -59.4453430, 2.3613091, -59.4538116, 2.4447346, -55.2343292, 55.1763229
38: -50.8822632, 8.5882998, -50.9295578, 8.7075768, -59.5898399, 59.5178566
39: -52.1420403, 14.8695374, -52.1801300, 14.9313087, -67.0733490, 67.0496674
40: -47.8287048, 8.3266706, -47.8784065, 8.4336395, -53.2490692, 53.1720352
41: -31.8296127, 15.2351923, -31.8973656, 15.3788185, -45.5982666, 45.5130920
42: -27.1431198, 10.0647602, -27.2015858, 10.2284002, -36.8211060, 36.6852264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=178, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7262839, upper bound: 14.8130237
time: 33.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7711206, upper bound: 14.8130237
time: 34.85 seconds

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

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237
time: 38.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 28.30 seconds

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

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7204540, upper bound: 14.8130237
time: 46.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 35.66 seconds

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

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7391718, upper bound: 14.8130237
time: 39.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 42.87 seconds

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

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7682095, upper bound: 14.8130237
time: 41.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237
time: 34.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 78.30 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6951250, upper bound: 14.8130237
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6972172, upper bound: 14.8130237
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7420616, upper bound: 14.8130237
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7262839, upper bound: 14.8130237
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7711206, upper bound: 14.8130237
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7204540, upper bound: 14.8130237
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7391718, upper bound: 14.8130237
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7682095, upper bound: 14.8130237
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -23.5435085, 32.7309723, -23.7919216, 32.8518982, -53.9618149, 54.8180695
1: -7.4647827, 32.2671471, -7.6064978, 32.3109741, -36.0519791, 36.3505135
2: -4.4838681, 31.6635532, -4.6639690, 31.7542610, -32.7754364, 32.9321289
3: -8.6969795, 28.7972107, -8.8645267, 28.8953724, -31.8409958, 32.2753830
4: -9.7928839, 34.9393349, -10.0022602, 35.0361710, -43.0700073, 43.3147888
5: -10.8459625, 29.7682686, -11.0483313, 29.9020939, -37.8624115, 38.0744400
6: -38.5739708, 7.3252058, -38.6448097, 7.4264441, -43.4524002, 44.2157745
7: -14.9855566, 30.5463600, -15.1929216, 30.6017075, -40.2087860, 41.5153961
8: -15.1264277, 34.3991890, -15.3420744, 34.5277176, -47.0354767, 46.7984161
9: -10.1158953, 27.0009460, -10.2751865, 27.1396561, -35.5277481, 35.9331169
10: -28.1219826, 23.5706024, -28.4010201, 23.8151550, -50.0743713, 50.1254959
11: -35.4800797, 14.0074425, -35.7538757, 14.1869965, -49.5088196, 49.1104279
12: -49.1530075, 1.4623423, -49.3777161, 1.8245854, -43.4939194, 43.5184326
13: -28.7354946, 21.1599503, -28.7910519, 21.2825851, -49.4329910, 49.9510040
14: -70.7025146, -6.6836662, -70.9236298, -6.5039864, -64.1985321, 64.2399597
15: -17.2311420, 24.5264130, -17.3574791, 24.6606255, -41.8917694, 41.8838921
16: -27.1850262, 23.4690819, -27.4375267, 23.6090317, -47.5873108, 48.2112045
17: -71.0323868, -4.1184120, -71.1203232, -3.9824753, -67.0499115, 67.0019073
18: -34.6373291, 11.6065302, -34.7255440, 11.6517591, -40.4371872, 40.2909317
19: -25.5759010, 5.1869898, -25.6792793, 5.2351050, -29.7857819, 29.6180801
20: -26.3381863, 4.2457690, -26.4541512, 4.3262081, -29.3038254, 28.9754181
21: -31.1554012, 9.9771271, -31.3350639, 10.0850830, -40.2448883, 40.2201233
22: -33.5077095, 6.8334746, -33.5940094, 6.9318557, -38.2933655, 36.7030106
23: -26.7825737, 8.7811632, -26.8946857, 8.8485832, -35.1161880, 34.8172455
24: -23.1397800, 9.8189802, -23.2218838, 9.8364305, -32.6630173, 32.4485741
25: -29.1067467, 5.9445391, -29.1938648, 6.0377407, -34.2283249, 32.7898254
26: -42.7778702, 7.5555172, -42.9243813, 7.7479992, -43.6844482, 43.4555740
27: -26.5097904, 11.4227467, -26.5872593, 11.4686604, -37.9784508, 37.6426849
28: -29.5188370, 7.0632467, -29.5786209, 7.1305456, -36.4996948, 36.3436203
29: -32.5128632, 8.8555470, -32.5829201, 8.9569607, -41.4698257, 41.4384689
30: -37.4768791, 6.8268719, -37.5717010, 6.9229975, -44.3998756, 44.3985748
31: -31.2397194, 7.2062116, -31.3796654, 7.2399874, -37.4461136, 37.0603638
32: -33.5294113, 6.5498037, -33.6271896, 6.6805849, -39.9247360, 40.1769943
33: -43.7108192, 15.8172817, -43.8977737, 15.9678555, -56.8810806, 56.9702530
34: -50.5612221, -4.4343901, -50.6533775, -4.3026018, -42.0827560, 42.0568390
35: -40.7730103, 6.8669853, -40.9121704, 7.0209041, -43.6562805, 43.7646255
36: -44.3465881, 5.2529173, -44.3903427, 5.3725195, -45.3378677, 45.3765640
37: -59.2907982, 2.2604852, -59.3916016, 2.3365545, -54.9509888, 55.3933716
38: -50.7408218, 8.4416866, -50.8379517, 8.5605812, -59.3014030, 59.2796402
39: -52.0045586, 14.7939482, -52.1144104, 14.8770800, -66.8816376, 66.9083557
40: -47.6948700, 8.2524614, -47.7915878, 8.3031197, -52.8962326, 53.0109634
41: -31.7025928, 15.1356020, -31.8004856, 15.2162848, -44.9396210, 45.3241577
42: -27.0134773, 9.9494762, -27.1334190, 10.0950003, -36.4777832, 36.4998627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
time: 37.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 34.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -23.7574425, 32.7889061, -23.8850212, 32.8551865, -54.1741791, 54.2094803
1: -7.6040649, 32.3092766, -7.6676140, 32.3134689, -36.4085617, 36.4586830
2: -4.6166458, 31.7011967, -4.7225018, 31.7575417, -32.9936829, 33.0332909
3: -8.7968626, 28.8311729, -8.9080029, 28.8997574, -32.3046341, 32.3609390
4: -9.9708614, 34.9879456, -10.0796337, 35.0395279, -43.2382965, 43.3026962
5: -10.9611607, 29.8137398, -11.0986004, 29.9077816, -37.9658051, 38.0142517
6: -38.6171989, 7.4258060, -38.6514359, 7.4694624, -44.2636795, 44.3124161
7: -15.1715431, 30.5946579, -15.2745953, 30.6040077, -41.5624695, 41.6575775
8: -15.3047028, 34.4547310, -15.4194298, 34.5333176, -46.9132538, 46.9389572
9: -10.2559280, 27.0454082, -10.3361254, 27.1436806, -35.6514206, 35.6617966
10: -28.2271576, 23.6272202, -28.4464684, 23.8244743, -50.2266998, 50.2284317
11: -35.5131531, 14.0477657, -35.7676506, 14.1976728, -49.0562363, 49.1696167
12: -49.2050705, 1.6363997, -49.3825645, 1.8983917, -43.7469025, 43.6438599
13: -28.7735443, 21.1921196, -28.8067913, 21.2947998, -49.4836273, 49.4171143
14: -70.8421173, -6.6292629, -70.9854279, -6.4960327, -64.3460846, 64.3561630
15: -17.3058147, 24.5577812, -17.3881474, 24.6657124, -41.9715271, 41.9459305
16: -27.3423004, 23.5199280, -27.5061169, 23.6143150, -48.2848358, 48.3344040
17: -71.0819550, -4.0848503, -71.1409760, -3.9758453, -67.1061096, 67.0561218
18: -34.6623383, 11.6367760, -34.7333984, 11.6638489, -40.4642830, 40.5118484
19: -25.6100292, 5.2007856, -25.6896629, 5.2409220, -29.8359146, 29.8562737
20: -26.3632908, 4.2621269, -26.4609795, 4.3338952, -28.9891891, 28.9923477
21: -31.1996136, 9.9958000, -31.3500118, 10.0924606, -40.1786270, 40.2468567
22: -33.5565567, 6.8925295, -33.6057091, 6.9576941, -38.3700562, 38.3717804
23: -26.8118362, 8.8174114, -26.9024582, 8.8636703, -35.1724930, 35.1832504
24: -23.1649399, 9.8367500, -23.2310066, 9.8455372, -32.7012711, 32.7410965
25: -29.1488972, 6.0216289, -29.2028961, 6.0718193, -34.3066940, 34.2795601
26: -42.8224716, 7.6032643, -42.9338188, 7.7687788, -43.5254593, 43.4986877
27: -26.5404987, 11.4595413, -26.5978622, 11.4796715, -37.6455002, 37.6951218
28: -29.5560970, 7.1318240, -29.5861702, 7.1604123, -36.5687637, 36.5519485
29: -32.5536270, 8.9151516, -32.5934906, 8.9831238, -41.5367508, 41.5086441
30: -37.4989929, 6.8635035, -37.5804825, 6.9366875, -44.4356804, 44.4439850
31: -31.2852459, 7.2365184, -31.3912277, 7.2524357, -37.5342560, 37.5466232
32: -33.5718193, 6.6406879, -33.6344223, 6.7208338, -40.2926521, 40.2751083
33: -43.7876434, 15.9067326, -43.9153900, 16.0077267, -56.9993362, 57.0176163
34: -50.6120529, -4.3227129, -50.6600494, -4.2537155, -42.1966171, 42.1682739
35: -40.8321571, 6.9773088, -40.9217033, 7.0695963, -43.7769623, 43.7496567
36: -44.4021950, 5.3787289, -44.3976326, 5.4279318, -45.4576111, 45.3985672
37: -59.3588486, 2.3267679, -59.4070244, 2.3654461, -55.0588913, 55.0417404
38: -50.8154564, 8.5468102, -50.8490829, 8.6058569, -59.4213142, 59.3958931
39: -52.0825005, 14.8172007, -52.1352234, 14.8867817, -66.9692841, 66.9524231
40: -47.7536659, 8.2855988, -47.8055649, 8.3168163, -53.0534592, 53.0531921
41: -31.7511139, 15.2105227, -31.8110237, 15.2482796, -45.3703918, 45.4017105
42: -27.0493088, 10.0381107, -27.1392384, 10.1328344, -36.5739288, 36.5842285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
time: 31.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 32.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -23.5456924, 32.7382240, -23.8654823, 32.8818741, -53.9952393, 54.8973541
1: -7.4657373, 32.2751694, -7.6577063, 32.3386116, -36.0769577, 36.4112930
2: -4.4857020, 31.6794395, -4.7566185, 31.8030014, -32.8237762, 33.0406342
3: -8.6988707, 28.7995491, -8.9125576, 28.9117584, -31.8562241, 32.3287277
4: -9.7945127, 34.9546814, -10.0943451, 35.0861664, -43.1205292, 43.4209366
5: -10.8478689, 29.7724915, -11.0944519, 29.9245358, -37.8847504, 38.1319122
6: -38.6015282, 7.3265877, -38.7326393, 7.5605154, -43.6133347, 44.3000412
7: -14.9870396, 30.5575581, -15.2532310, 30.6365833, -40.2436142, 41.5873184
8: -15.1283884, 34.4144592, -15.4241161, 34.5776176, -47.0862122, 46.8960190
9: -10.1133184, 27.0028648, -10.3040228, 27.1736870, -35.5618057, 35.9671059
10: -28.1240921, 23.5820770, -28.4801788, 23.8669758, -50.1276093, 50.2568741
11: -35.4814377, 14.0169621, -35.8014755, 14.2268190, -49.5474243, 49.1665039
12: -49.1629295, 1.4642167, -49.4161377, 1.8684616, -43.5508270, 43.5624542
13: -28.7384682, 21.1603088, -28.8324966, 21.3100891, -49.4605713, 49.9928055
14: -70.7045898, -6.6583672, -71.0647888, -6.4285908, -64.2760010, 64.4064178
15: -17.2325916, 24.5285587, -17.3911953, 24.6772881, -41.9098816, 41.9197540
16: -27.1861515, 23.4721680, -27.4951363, 23.6356621, -47.6157990, 48.2730865
17: -71.0372620, -4.0783005, -71.3112411, -3.8693275, -67.1679382, 67.2329407
18: -34.6389427, 11.6049709, -34.7684669, 11.6754436, -40.4677010, 40.3284035
19: -25.5774593, 5.1900940, -25.7128124, 5.2582064, -29.8089447, 29.6478233
20: -26.3383198, 4.2470012, -26.4646149, 4.3493147, -29.3234367, 28.9904938
21: -31.1562214, 9.9798679, -31.3654213, 10.1084213, -40.2682495, 40.2478867
22: -33.5096550, 6.8362370, -33.6290588, 6.9569902, -38.3193130, 36.7429047
23: -26.7838326, 8.7834387, -26.9212494, 8.8728676, -35.1425934, 34.8454590
24: -23.1409149, 9.8246574, -23.2787266, 9.8726110, -32.6991310, 32.5076828
25: -29.1070995, 5.9493055, -29.2427921, 6.0724897, -34.2652054, 32.8438263
26: -42.7795792, 7.5548477, -42.9501762, 7.7611179, -43.7043991, 43.4868011
27: -26.5185261, 11.4236279, -26.6333580, 11.5326557, -38.0511818, 37.6930618
28: -29.5195694, 7.0654078, -29.5971756, 7.1730042, -36.5481720, 36.3648148
29: -32.5138550, 8.8584576, -32.6140900, 8.9817486, -41.4956055, 41.4725494
30: -37.4767647, 6.8295803, -37.5862503, 6.9516001, -44.4283638, 44.4158325
31: -31.2420769, 7.2161164, -31.4183674, 7.2804337, -37.4828491, 37.0985260
32: -33.5566788, 6.5505695, -33.7174721, 6.8173723, -40.0900116, 40.2680435
33: -43.7134018, 15.8185759, -43.9219017, 16.0116272, -56.9416809, 56.9995117
34: -50.5781708, -4.4336395, -50.7152252, -4.2119923, -42.1989059, 42.1197052
35: -40.7816429, 6.8676734, -40.9495659, 7.0685439, -43.7198792, 43.7990799
36: -44.3743324, 5.2536769, -44.4781532, 5.4954567, -45.4888306, 45.4643631
37: -59.3018036, 2.2621207, -59.4385605, 2.4196420, -55.0556717, 55.4411621
38: -50.7616806, 8.4430246, -50.9189491, 8.6654854, -59.4271660, 59.3619728
39: -52.0122375, 14.7948523, -52.1587868, 14.9252224, -66.9374619, 66.9536362
40: -47.7169151, 8.2537928, -47.8637085, 8.4215250, -53.0377808, 53.0826187
41: -31.7276630, 15.1367283, -31.8866615, 15.3495150, -45.0984344, 45.4094467
42: -27.0311546, 9.9509058, -27.1959686, 10.1953182, -36.5953140, 36.5605850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6951250, upper bound: 14.7919458
time: 36.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6951250, upper bound: 14.8130237
time: 44.07 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -23.7596073, 32.7961502, -23.9586201, 32.8852386, -54.2075729, 54.2902451
1: -7.6050005, 32.3173294, -7.7188311, 32.3411636, -36.4335251, 36.5195084
2: -4.6184998, 31.7170792, -4.8151302, 31.8062534, -33.0417633, 33.1417313
3: -8.7987556, 28.8335419, -8.9560394, 28.9161301, -32.3204346, 32.4143486
4: -9.9724874, 35.0032692, -10.1717129, 35.0895233, -43.2888184, 43.4090271
5: -10.9630547, 29.8179550, -11.1447392, 29.9301987, -37.9881058, 38.0736084
6: -38.6448174, 7.4271450, -38.7392654, 7.6035929, -44.4248657, 44.3967056
7: -15.1730843, 30.6058750, -15.3349113, 30.6389217, -41.5977859, 41.7295151
8: -15.3066988, 34.4699745, -15.5015192, 34.5832176, -46.9619904, 47.0365372
9: -10.2533388, 27.0473537, -10.3649626, 27.1777191, -35.6854858, 35.6972885
10: -28.2293301, 23.6386452, -28.5256996, 23.8763046, -50.2802734, 50.3598633
11: -35.5145493, 14.0573025, -35.8152657, 14.2374554, -49.0983734, 49.2257080
12: -49.2149734, 1.6382594, -49.4209900, 1.9422731, -43.8076248, 43.6878510
13: -28.7765007, 21.1925735, -28.8482361, 21.3223152, -49.5112457, 49.4689331
14: -70.8441620, -6.6039257, -71.1266098, -6.4206848, -64.4234772, 64.5226822
15: -17.3072128, 24.5599003, -17.4218998, 24.6824112, -41.9896240, 41.9818001
16: -27.3434486, 23.5230179, -27.5637398, 23.6409225, -48.3135834, 48.3962784
17: -71.0868683, -4.0447388, -71.3319092, -3.8627663, -67.2241058, 67.2871704
18: -34.6639633, 11.6352539, -34.7763443, 11.6875467, -40.4947853, 40.5508499
19: -25.6116142, 5.2039404, -25.7232513, 5.2640033, -29.8591461, 29.8871155
20: -26.3634434, 4.2633567, -26.4714661, 4.3569832, -29.0109596, 29.0074120
21: -31.2004128, 9.9985008, -31.3803673, 10.1158447, -40.2025299, 40.2746506
22: -33.5584908, 6.8953581, -33.6408424, 6.9828229, -38.3960114, 38.4294586
23: -26.8131027, 8.8197069, -26.9290085, 8.8879395, -35.1989136, 35.2112961
24: -23.1660824, 9.8424149, -23.2878113, 9.8816824, -32.7373657, 32.8017159
25: -29.1492290, 6.0264306, -29.2518654, 6.1065645, -34.3435669, 34.3395004
26: -42.8241463, 7.6025896, -42.9596291, 7.7818689, -43.5462189, 43.5298920
27: -26.5492706, 11.4604483, -26.6439190, 11.5436678, -37.7386627, 37.7454796
28: -29.5568619, 7.1339707, -29.6047153, 7.2028809, -36.6172867, 36.5727234
29: -32.5546265, 8.9180756, -32.6246948, 9.0078983, -41.5625229, 41.5427704
30: -37.4989128, 6.8662548, -37.5950089, 6.9652786, -44.4641914, 44.4612656
31: -31.2876053, 7.2464347, -31.4298916, 7.2928524, -37.5709839, 37.5862198
32: -33.5990448, 6.6414871, -33.7246933, 6.8576298, -40.4566727, 40.3661804
33: -43.7902412, 15.9080238, -43.9395142, 16.0515442, -57.0599518, 57.0504074
34: -50.6290131, -4.3219166, -50.7218857, -4.1630988, -42.3127518, 42.2308731
35: -40.8407707, 6.9780254, -40.9590912, 7.1172423, -43.8406143, 43.7856140
36: -44.4299240, 5.3794384, -44.4854279, 5.5508623, -45.6084442, 45.4862213
37: -59.3698578, 2.3283682, -59.4540100, 2.4485769, -55.1635437, 55.0912857
38: -50.8362961, 8.5481567, -50.9300766, 8.7107735, -59.5470695, 59.4782333
39: -52.0902176, 14.8181429, -52.1795654, 14.9349203, -67.0251389, 66.9977112
40: -47.7756577, 8.2869244, -47.8777580, 8.4352427, -53.1939621, 53.1248322
41: -31.7761955, 15.2116060, -31.8971939, 15.3815432, -45.5298615, 45.4869461
42: -27.0669956, 10.0395098, -27.2017899, 10.2331848, -36.6917725, 36.6448898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7399635, upper bound: 14.7919458
time: 53.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7399635, upper bound: 14.8130237
time: 49.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.6287079, 32.8806686, -23.7951679, 32.8522949, -54.0494080, 54.2168961
1: -7.4981365, 32.3271713, -7.6060610, 32.3106880, -36.3155212, 36.4059219
2: -4.5446854, 31.7924213, -4.6698580, 31.7537708, -32.9305725, 33.0760040
3: -8.7680941, 28.9737396, -8.8721027, 28.8947792, -32.2861710, 32.4525642
4: -9.8395433, 35.0467491, -10.0054283, 35.0356369, -43.1171417, 43.2883987
5: -10.9262161, 29.9576530, -11.0573931, 29.9018784, -37.9379578, 38.1100159
6: -38.6270256, 7.3437405, -38.6446533, 7.4230862, -44.2569427, 44.2335663
7: -15.0579157, 30.6457500, -15.1950674, 30.6004181, -41.4755096, 41.6237717
8: -15.1745453, 34.5156364, -15.3442764, 34.5274010, -46.7912292, 46.9323807
9: -10.1784630, 27.0380440, -10.2742014, 27.1382408, -35.5827942, 35.5944328
10: -28.3359241, 23.6558323, -28.4010239, 23.8221035, -50.3421783, 50.2088394
11: -35.7018204, 14.0601902, -35.7535934, 14.1919317, -49.2422256, 49.1608505
12: -49.4326706, 1.5841122, -49.3762665, 1.8362260, -43.9037628, 43.6342926
13: -28.7580986, 21.2621117, -28.7870884, 21.2839108, -49.4563751, 49.4389343
14: -70.9055023, -6.6137428, -70.9224472, -6.4968700, -64.4086304, 64.3087006
15: -17.2486725, 24.6091690, -17.3477554, 24.6607666, -41.9094391, 41.9569244
16: -27.2799606, 23.4795761, -27.4367580, 23.6053581, -48.1910858, 48.2209244
17: -71.1796570, -4.0391884, -71.1192627, -3.9733715, -67.2062836, 67.0800781
18: -34.8001785, 11.6374664, -34.7252541, 11.6517096, -40.6100502, 40.5037918
19: -25.6797810, 5.2061310, -25.6795788, 5.2371655, -29.8726501, 29.8465347
20: -26.4475689, 4.2766814, -26.4546852, 4.3284297, -29.0403633, 29.0080299
21: -31.3293533, 10.0198460, -31.3349953, 10.0897503, -40.3037415, 40.2627563
22: -33.5844803, 6.8765483, -33.5926590, 6.9292645, -38.3763199, 38.3560333
23: -26.8898849, 8.8061543, -26.8946743, 8.8499842, -35.2068100, 35.1659317
24: -23.2041073, 9.8342352, -23.2217216, 9.8340845, -32.7136879, 32.7301407
25: -29.1736069, 5.9761477, -29.1932163, 6.0396142, -34.2782593, 34.2293129
26: -43.0537758, 7.6626081, -42.9231758, 7.7570686, -43.7474747, 43.5593643
27: -26.5867958, 11.4398479, -26.5879021, 11.4676552, -37.6722794, 37.6610641
28: -29.5842476, 7.0887160, -29.5789356, 7.1296616, -36.5529633, 36.5151749
29: -32.5876236, 8.8917561, -32.5816498, 8.9587336, -41.5463562, 41.4734039
30: -37.5872498, 6.8658457, -37.5715179, 6.9245424, -44.5117912, 44.4373627
31: -31.3724537, 7.2215271, -31.3802834, 7.2410250, -37.5320244, 37.5321503
32: -33.6086655, 6.5784817, -33.6267929, 6.6795855, -40.2882500, 40.2052765
33: -43.7643204, 15.9296017, -43.9017448, 15.9676275, -56.9343567, 57.0560532
34: -50.5928764, -4.3527746, -50.6554260, -4.3027821, -42.1112671, 42.1696243
35: -40.8096695, 6.9672432, -40.9149284, 7.0211062, -43.6922302, 43.7693481
36: -44.3632202, 5.2905588, -44.3896255, 5.3705606, -45.3566895, 45.3343124
37: -59.3651390, 2.2914901, -59.3911743, 2.3324823, -55.0194855, 55.0617828
38: -50.7859688, 8.4805746, -50.8372993, 8.5572109, -59.3431778, 59.3178749
39: -52.0550156, 14.8439608, -52.1146774, 14.8732500, -66.9282684, 66.9586411
40: -47.7463112, 8.2914848, -47.7921181, 8.3013992, -53.0355988, 53.0531540
41: -31.7547150, 15.1568918, -31.8005161, 15.2130690, -45.3568573, 45.3429031
42: -27.0889454, 9.9733782, -27.1331902, 10.0900106, -36.6205978, 36.5374489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6972172, upper bound: 14.7898460
time: 37.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6972172, upper bound: 14.8130237
time: 37.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.8426208, 32.9393082, -23.8882523, 32.8556137, -54.2617035, 54.3712082
1: -7.6375523, 32.3693619, -7.6671944, 32.3131981, -36.4477234, 36.5140915
2: -4.6775455, 31.8300381, -4.7283802, 31.7570610, -33.0530090, 33.1771278
3: -8.8681402, 29.0077629, -8.9155731, 28.8991318, -32.3680801, 32.5381432
4: -10.0174809, 35.0954590, -10.0827579, 35.0389900, -43.2854309, 43.4163437
5: -11.0413818, 30.0032272, -11.1076775, 29.9075031, -38.0413208, 38.2142715
6: -38.6709442, 7.4442644, -38.6512756, 7.4661598, -44.3509521, 44.3301926
7: -15.2444630, 30.6940365, -15.2767286, 30.6026878, -41.6313934, 41.7659454
8: -15.3533669, 34.5711517, -15.4216356, 34.5330276, -46.9644012, 47.0729370
9: -10.3184299, 27.0830097, -10.3351431, 27.1422653, -35.7063675, 35.7047348
10: -28.4412098, 23.7125549, -28.4464951, 23.8314362, -50.4482346, 50.3118591
11: -35.7360458, 14.1004963, -35.7673645, 14.2026110, -49.2872849, 49.2200470
12: -49.4849472, 1.7580810, -49.3810730, 1.9100380, -44.0455017, 43.7596130
13: -28.7961655, 21.2949390, -28.8028297, 21.2961826, -49.5070496, 49.5104218
14: -71.0451126, -6.5592766, -70.9842758, -6.4889069, -64.5562057, 64.4250031
15: -17.3232574, 24.6412506, -17.3784237, 24.6658878, -41.9891434, 42.0196762
16: -27.4375839, 23.5305271, -27.5053616, 23.6106186, -48.3361053, 48.3441772
17: -71.2292099, -4.0050430, -71.1399155, -3.9668312, -67.2623749, 67.1348724
18: -34.8251953, 11.6679611, -34.7331429, 11.6638222, -40.6371613, 40.5552139
19: -25.7139435, 5.2201042, -25.6899948, 5.2429633, -29.9228592, 29.8716202
20: -26.4731464, 4.2930322, -26.4614792, 4.3361044, -29.1059189, 29.0249252
21: -31.3737392, 10.0386190, -31.3499470, 10.0971518, -40.3677521, 40.2895584
22: -33.6333313, 6.9363070, -33.6043777, 6.9551320, -38.4529800, 38.4158859
23: -26.9191704, 8.8425694, -26.9024620, 8.8650694, -35.2631607, 35.2129288
24: -23.2292747, 9.8521595, -23.2308044, 9.8431854, -32.7519455, 32.7576218
25: -29.2157841, 6.0538559, -29.2022667, 6.0736923, -34.3566284, 34.3133469
26: -43.0985947, 7.7103033, -42.9325790, 7.7778244, -43.8140564, 43.6024399
27: -26.6179390, 11.4766865, -26.5985012, 11.4786453, -37.7190552, 37.7134590
28: -29.6215763, 7.1573772, -29.5864582, 7.1595654, -36.6220703, 36.5919952
29: -32.6283493, 8.9517994, -32.5922623, 8.9849052, -41.6132545, 41.5440598
30: -37.6101074, 6.9025917, -37.5802231, 6.9382057, -44.5483131, 44.4828148
31: -31.4179382, 7.2521620, -31.3918190, 7.2534642, -37.6202240, 37.5745773
32: -33.6515274, 6.6693544, -33.6340103, 6.7198210, -40.3713493, 40.3033638
33: -43.8412170, 16.0198917, -43.9193497, 16.0074692, -57.0526276, 57.1484833
34: -50.6437149, -4.2410765, -50.6621246, -4.2538943, -42.2251282, 42.2488403
35: -40.8688278, 7.0779548, -40.9243965, 7.0697846, -43.8129578, 43.8492508
36: -44.4188347, 5.4164100, -44.3969574, 5.4259324, -45.4763412, 45.4411469
37: -59.4332047, 2.3581915, -59.4065857, 2.3613811, -55.1274109, 55.0920792
38: -50.8607101, 8.5856571, -50.8484039, 8.6024656, -59.4631767, 59.4340591
39: -52.1329765, 14.8677073, -52.1354942, 14.8829851, -67.0159607, 67.0032043
40: -47.8058891, 8.3245993, -47.8061142, 8.3151169, -53.1070404, 53.0953827
41: -31.8036671, 15.2317495, -31.8110447, 15.2450886, -45.4375153, 45.4204102
42: -27.1251583, 10.0619411, -27.1389790, 10.1278543, -36.7132874, 36.6217804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7420616, upper bound: 14.7898460
time: 34.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7420616, upper bound: 14.8130237
time: 21.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.6308899, 32.8879471, -23.8687286, 32.8823624, -54.0828323, 54.2975464
1: -7.4990931, 32.3352203, -7.6572733, 32.3383751, -36.3405075, 36.4667511
2: -4.5465441, 31.8083172, -4.7624979, 31.8024693, -32.9787140, 33.1844406
3: -8.7700024, 28.9761372, -8.9201250, 28.9111423, -32.3020248, 32.5059433
4: -9.8411808, 35.0620956, -10.0974712, 35.0856247, -43.1677017, 43.3946457
5: -10.9281178, 29.9618378, -11.1034994, 29.9243202, -37.9602737, 38.1693802
6: -38.6546631, 7.3451033, -38.7324677, 7.5571814, -44.4181213, 44.3178787
7: -15.0594568, 30.6569767, -15.2554035, 30.6352844, -41.5108795, 41.6957016
8: -15.1765242, 34.5308647, -15.4262981, 34.5773849, -46.8399582, 47.0299759
9: -10.1758862, 27.0399742, -10.3030519, 27.1723003, -35.6168213, 35.6299591
10: -28.3380413, 23.6672401, -28.4801769, 23.8739128, -50.3957214, 50.3402557
11: -35.7031898, 14.0696583, -35.8011856, 14.2317257, -49.2844238, 49.2168961
12: -49.4425888, 1.5859361, -49.4147110, 1.8800931, -43.9644623, 43.6783218
13: -28.7611046, 21.2625427, -28.8285389, 21.3114319, -49.4839401, 49.4906921
14: -70.9075928, -6.5884094, -71.0636139, -6.4214954, -64.4860992, 64.4752045
15: -17.2500992, 24.6113358, -17.3815117, 24.6774521, -41.9275513, 41.9928474
16: -27.2811069, 23.4827023, -27.4943676, 23.6318645, -48.2198563, 48.2827301
17: -71.1845245, -3.9990845, -71.3101959, -3.8602676, -67.3242569, 67.3111115
18: -34.8018150, 11.6359291, -34.7682114, 11.6754293, -40.6405754, 40.5427551
19: -25.6813755, 5.2092438, -25.7131195, 5.2602720, -29.8958511, 29.8773689
20: -26.4477158, 4.2779260, -26.4651546, 4.3515172, -29.0621414, 29.0230865
21: -31.3301811, 10.0225563, -31.3653603, 10.1131649, -40.3276367, 40.2905579
22: -33.5864296, 6.8793187, -33.6277237, 6.9544487, -38.4022369, 38.4135818
23: -26.8911514, 8.8084383, -26.9212265, 8.8742599, -35.2331924, 35.1939392
24: -23.2052422, 9.8399258, -23.2785263, 9.8702717, -32.7498169, 32.7907715
25: -29.1739845, 5.9809475, -29.2421665, 6.0743909, -34.3151703, 34.2892227
26: -43.0554733, 7.6619630, -42.9489746, 7.7702117, -43.7681427, 43.5905609
27: -26.5955563, 11.4407501, -26.6340313, 11.5316133, -37.7654190, 37.7113724
28: -29.5849915, 7.0908499, -29.5974846, 7.1721354, -36.6014862, 36.5359955
29: -32.5886459, 8.8946743, -32.6128082, 8.9835224, -41.5721664, 41.5074844
30: -37.5871277, 6.8685837, -37.5860367, 6.9531307, -44.5402603, 44.4546204
31: -31.3748531, 7.2314572, -31.4189301, 7.2814517, -37.5687561, 37.5717468
32: -33.6359138, 6.5792599, -33.7170944, 6.8164110, -40.4523239, 40.2963562
33: -43.7669106, 15.9309978, -43.9258919, 16.0113945, -56.9949036, 57.0889359
34: -50.6098480, -4.3519535, -50.7173195, -4.2121701, -42.2274246, 42.2322540
35: -40.8182716, 6.9679499, -40.9523201, 7.0687666, -43.7558823, 43.8053360
36: -44.3910027, 5.2913156, -44.4775391, 5.4935026, -45.5075989, 45.4220200
37: -59.3761292, 2.2931237, -59.4381104, 2.4155574, -55.1241455, 55.1112442
38: -50.8067245, 8.4819918, -50.9182968, 8.6621513, -59.4688759, 59.4002876
39: -52.0627289, 14.8448610, -52.1590805, 14.9214020, -66.9841309, 67.0039444
40: -47.7683372, 8.2927904, -47.8642540, 8.4197922, -53.1760712, 53.1248093
41: -31.7798367, 15.1580048, -31.8867397, 15.3463173, -45.5163422, 45.4281693
42: -27.1065788, 9.9747896, -27.1957188, 10.1903057, -36.7384033, 36.5981750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7262839, upper bound: 14.7898460
time: 37.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7262839, upper bound: 14.8130237
time: 83.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.8448009, 32.9465561, -23.9618626, 32.8856735, -54.2951050, 54.4519577
1: -7.6385298, 32.3773956, -7.7183867, 32.3409004, -36.4727097, 36.5749207
2: -4.6794014, 31.8459244, -4.8210278, 31.8057346, -33.1011047, 33.2855835
3: -8.8700428, 29.0100975, -8.9636278, 28.9155064, -32.3839264, 32.5915489
4: -10.0191402, 35.1108093, -10.1748590, 35.0889778, -43.3359528, 43.5226288
5: -11.0432577, 30.0074749, -11.1537800, 29.9299545, -38.0636139, 38.2737122
6: -38.6985550, 7.4456148, -38.7390976, 7.6002283, -44.5120850, 44.4144745
7: -15.2460032, 30.7052479, -15.3370380, 30.6376057, -41.6667328, 41.8378601
8: -15.3553410, 34.5864143, -15.5036917, 34.5829849, -47.0130768, 47.1705399
9: -10.3158655, 27.0849419, -10.3639612, 27.1762905, -35.7404327, 35.7402420
10: -28.4433041, 23.7239609, -28.5256958, 23.8832703, -50.5018082, 50.4432449
11: -35.7374420, 14.1100473, -35.8149719, 14.2423515, -49.3294373, 49.2761917
12: -49.4948959, 1.7599149, -49.4195480, 1.9539390, -44.1062088, 43.8036270
13: -28.7991905, 21.2953606, -28.8443012, 21.3236656, -49.5346909, 49.5622330
14: -71.0472260, -6.5339088, -71.1254120, -6.4135513, -64.6336746, 64.5915070
15: -17.3247032, 24.6433983, -17.4121819, 24.6825752, -42.0072784, 42.0555801
16: -27.4387398, 23.5336304, -27.5629139, 23.6371002, -48.3649139, 48.4060059
17: -71.2341309, -3.9649734, -71.3308182, -3.8536854, -67.3804474, 67.3658447
18: -34.8268051, 11.6664600, -34.7761154, 11.6875238, -40.6676598, 40.5942001
19: -25.7155037, 5.2232428, -25.7235336, 5.2660627, -29.9460907, 29.9024582
20: -26.4732990, 4.2942748, -26.4719849, 4.3592100, -29.1277084, 29.0400085
21: -31.3745937, 10.0413408, -31.3803196, 10.1205645, -40.3916321, 40.3173676
22: -33.6352654, 6.9391332, -33.6394882, 6.9802766, -38.4789200, 38.4735413
23: -26.9204159, 8.8448753, -26.9289913, 8.8893661, -35.2895889, 35.2409096
24: -23.2304173, 9.8578224, -23.2876434, 9.8793449, -32.7880249, 32.8182602
25: -29.2161217, 6.0586343, -29.2512341, 6.1084323, -34.3935394, 34.3732719
26: -43.1002846, 7.7096410, -42.9583626, 7.7909889, -43.8347931, 43.6336594
27: -26.6267300, 11.4775934, -26.6445732, 11.5425997, -37.8122177, 37.7638283
28: -29.6223068, 7.1594968, -29.6050110, 7.2020302, -36.6705780, 36.6127396
29: -32.6294174, 8.9547062, -32.6234512, 9.0096798, -41.6390991, 41.5781555
30: -37.6100197, 6.9053535, -37.5947800, 6.9667845, -44.5768051, 44.5001335
31: -31.4203033, 7.2620621, -31.4305077, 7.2938700, -37.6569252, 37.6141891
32: -33.6787376, 6.6701479, -33.7243042, 6.8566885, -40.5354271, 40.3944511
33: -43.8438377, 16.0212421, -43.9434891, 16.0512638, -57.1132736, 57.1813507
34: -50.6606903, -4.2402763, -50.7239609, -4.1632833, -42.3412857, 42.3114319
35: -40.8774643, 7.0787134, -40.9618301, 7.1174884, -43.8766022, 43.8851929
36: -44.4465408, 5.4171438, -44.4848022, 5.5489097, -45.6271820, 45.5288620
37: -59.4441872, 2.3598638, -59.4535561, 2.4444995, -55.2320862, 55.1415863
38: -50.8815079, 8.5870695, -50.9294052, 8.7073469, -59.5888557, 59.5164757
39: -52.1407051, 14.8686666, -52.1798706, 14.9311676, -67.0718689, 67.0485382
40: -47.8279457, 8.3259153, -47.8782921, 8.4335108, -53.2474823, 53.1669922
41: -31.8287468, 15.2328167, -31.8972225, 15.3783607, -45.5969620, 45.5056458
42: -27.1428337, 10.0633316, -27.2015438, 10.2281532, -36.8310852, 36.6824951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7711206, upper bound: 14.7898460
time: 41.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7711206, upper bound: 14.8130237
time: 40.38 seconds

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

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
time: 58.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 36.22 seconds

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

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
time: 54.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 25.30 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -23.6854362, 32.8309174, -23.9158306, 32.8986588, -54.1208496, 54.2826080
1: -7.5359058, 32.2962265, -7.6829815, 32.3404884, -36.3872757, 36.4462814
2: -4.6100779, 31.7608261, -4.8115668, 31.8053036, -32.9879303, 33.1833115
3: -8.8115768, 28.8798981, -8.9623613, 28.9220428, -32.2994080, 32.4341240
4: -9.9339342, 35.0332031, -10.1556969, 35.0924721, -43.2268677, 43.4031296
5: -11.0024805, 29.8852882, -11.1634874, 29.9398575, -37.9871521, 38.1518250
6: -38.6903763, 7.3691902, -38.7598076, 7.5758972, -44.5349731, 44.3951797
7: -15.0878849, 30.5876808, -15.2881985, 30.6444321, -41.5311890, 41.6539993
8: -15.2713089, 34.5228653, -15.4892483, 34.5880203, -46.8874207, 47.0698700
9: -10.2003679, 27.1470299, -10.3081913, 27.2382126, -35.7179260, 35.6486549
10: -28.3542480, 23.8809090, -28.4898319, 24.0090160, -50.5487823, 50.3961945
11: -35.7522888, 14.2471380, -35.8282623, 14.3393192, -49.4321289, 49.3552017
12: -49.3618317, 1.7805281, -49.4269485, 2.0152025, -44.0162201, 43.8223724
13: -28.7728157, 21.2957401, -28.8396835, 21.3593216, -49.5547485, 49.5438080
14: -70.8557434, -6.4452114, -71.0750732, -6.3300610, -64.5256805, 64.6298599
15: -17.3426704, 24.6531124, -17.4378166, 24.6986351, -42.0413055, 42.0909271
16: -27.3612347, 23.6222820, -27.5144539, 23.7090855, -48.3879166, 48.3590546
17: -71.1100998, -3.8723755, -71.3235245, -3.7734680, -67.3366318, 67.4511490
18: -34.7253304, 11.6240330, -34.7860527, 11.6817408, -40.5746460, 40.5607452
19: -25.6706066, 5.2466769, -25.7309361, 5.2862854, -29.8871460, 29.9297943
20: -26.4471092, 4.3420181, -26.4807129, 4.3930192, -29.0695419, 29.0376816
21: -31.3178062, 10.1106339, -31.3836861, 10.1718826, -40.3826141, 40.2939987
22: -33.5666656, 6.9049144, -33.6478348, 6.9830723, -38.4018173, 38.4528580
23: -26.8828583, 8.8529930, -26.9326210, 8.9033070, -35.2067947, 35.2180252
24: -23.2314415, 9.8427258, -23.3113117, 9.8759327, -32.7929688, 32.8672981
25: -29.1712685, 6.0221672, -29.2582855, 6.1033421, -34.3150558, 34.3603210
26: -42.9065094, 7.7188473, -42.9718895, 7.8327999, -43.6977386, 43.6282501
27: -26.6107864, 11.4370461, -26.6675053, 11.5362473, -37.7993469, 37.7510796
28: -29.5629215, 7.0981522, -29.6110649, 7.1815271, -36.5926056, 36.5810318
29: -32.5635071, 8.9353275, -32.6238403, 9.0157852, -41.5792923, 41.5591660
30: -37.5672836, 6.9297457, -37.6037102, 6.9935904, -44.5608749, 44.5334549
31: -31.3645668, 7.2551627, -31.4395771, 7.2988310, -37.5364456, 37.6840782
32: -33.6704330, 6.6640720, -33.7304764, 6.8678398, -40.5382729, 40.3945465
33: -43.8855133, 15.9304104, -43.9995384, 16.0267315, -57.0767365, 57.1474609
34: -50.6791000, -4.3601618, -50.7602463, -4.2054453, -42.2702332, 42.2637253
35: -40.9201508, 6.9661336, -41.0136642, 7.0776939, -43.8201065, 43.8666000
36: -44.4259720, 5.3054876, -44.4986382, 5.5096183, -45.5665436, 45.4573364
37: -59.3897362, 2.3085160, -59.4691620, 2.4339418, -55.1740570, 55.1757355
38: -50.8533478, 8.5139084, -50.9540787, 8.6899347, -59.5432816, 59.4679871
39: -52.0972786, 14.8803492, -52.1886406, 14.9536781, -67.0509567, 67.0689926
40: -47.8232689, 8.2909164, -47.9018898, 8.4295635, -53.2361908, 53.1568985
41: -31.8445797, 15.1895561, -31.9152565, 15.3717995, -45.5977859, 45.4881821
42: -27.1568947, 10.0942907, -27.2117538, 10.2629747, -36.8616180, 36.7502747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
time: 26.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 42.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -23.8994026, 32.8896141, -24.0089569, 32.9019508, -54.3332520, 54.4369965
1: -7.6753111, 32.3384132, -7.7440910, 32.3429909, -36.5194702, 36.5544891
2: -4.7429771, 31.7984924, -4.8700728, 31.8085899, -33.1103516, 33.2844543
3: -8.9116669, 28.9139328, -9.0058327, 28.9264488, -32.3813324, 32.5197334
4: -10.1118240, 35.0818710, -10.2330141, 35.0958023, -43.3950653, 43.5309982
5: -11.1177378, 29.9309807, -11.2137585, 29.9455528, -38.0905304, 38.2561798
6: -38.7343369, 7.4697018, -38.7664223, 7.6189556, -44.6289749, 44.4917603
7: -15.2744865, 30.6360054, -15.3699007, 30.6467438, -41.6870575, 41.7961960
8: -15.4500952, 34.5783653, -15.5666418, 34.5936317, -47.0605011, 47.2104263
9: -10.3403511, 27.1919918, -10.3691187, 27.2422428, -35.8416061, 35.7590294
10: -28.4594784, 23.9375935, -28.5352802, 24.0183678, -50.6549149, 50.4991379
11: -35.7865410, 14.2874794, -35.8420258, 14.3499622, -49.4772873, 49.4144135
12: -49.4141464, 1.9544272, -49.4317856, 2.0890031, -44.1579437, 43.9476242
13: -28.8109264, 21.3287621, -28.8554268, 21.3716087, -49.6054230, 49.6155319
14: -70.9953766, -6.3907814, -71.1369019, -6.3221340, -64.6732407, 64.7461243
15: -17.4172382, 24.6851997, -17.4684887, 24.7037449, -42.1209831, 42.1536865
16: -27.5188370, 23.6732216, -27.5830307, 23.7143517, -48.5330200, 48.4822998
17: -71.1596985, -3.8381386, -71.3441620, -3.7669220, -67.3927765, 67.5060272
18: -34.7504272, 11.6545563, -34.7939796, 11.6938534, -40.6017952, 40.6121521
19: -25.7047691, 5.2606859, -25.7413712, 5.2920704, -29.9374390, 29.9548531
20: -26.4727325, 4.3583889, -26.4875679, 4.4007168, -29.1351967, 29.0546036
21: -31.3622551, 10.1293030, -31.3986778, 10.1792850, -40.4466934, 40.3207703
22: -33.6155167, 6.9646425, -33.6596375, 7.0089335, -38.4785385, 38.5128326
23: -26.9121380, 8.8894196, -26.9403572, 8.9183865, -35.2631836, 35.2650070
24: -23.2566547, 9.8605785, -23.3204708, 9.8849964, -32.8312607, 32.8947296
25: -29.2134666, 6.0998664, -29.2673893, 6.1373906, -34.3935013, 34.4443283
26: -42.9514618, 7.7666154, -42.9813156, 7.8536229, -43.7644882, 43.6714096
27: -26.6419792, 11.4738388, -26.6781178, 11.5472403, -37.8462372, 37.8034019
28: -29.6002617, 7.1667914, -29.6185970, 7.2114625, -36.6617584, 36.6578369
29: -32.6043091, 8.9953842, -32.6344681, 9.0419064, -41.6462173, 41.6298523
30: -37.5901871, 6.9664774, -37.6124802, 7.0072355, -44.5974236, 44.5789566
31: -31.4101448, 7.2857842, -31.4511528, 7.3112354, -37.6246490, 37.7265434
32: -33.7132607, 6.7549019, -33.7376862, 6.9080782, -40.6213379, 40.4925880
33: -43.9624557, 16.0206032, -44.0171890, 16.0666084, -57.1950760, 57.2399445
34: -50.7299347, -4.2484703, -50.7668915, -4.1565561, -42.3840790, 42.3429718
35: -40.9793358, 7.0768509, -41.0232506, 7.1263871, -43.9408493, 43.9464951
36: -44.4816246, 5.4313622, -44.5058708, 5.5650043, -45.6862488, 45.5642090
37: -59.4578285, 2.3752179, -59.4846191, 2.4628510, -55.2820282, 55.2060242
38: -50.9282112, 8.6189241, -50.9652023, 8.7351971, -59.6634064, 59.5841255
39: -52.1753502, 14.9040251, -52.2094383, 14.9634209, -67.1387711, 67.1134644
40: -47.8828888, 8.3240309, -47.9159164, 8.4433270, -53.3078308, 53.1991196
41: -31.8935127, 15.2644291, -31.9257851, 15.4038029, -45.6784592, 45.5657272
42: -27.1931458, 10.1828833, -27.2175713, 10.3008022, -36.9542542, 36.8345871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
time: 75.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 35.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.7711830, 32.9752617, -23.8456898, 32.8690224, -54.1772690, 54.3658905
1: -7.5688400, 32.3485222, -7.6314449, 32.3125076, -36.4039383, 36.4455414
2: -4.6725845, 31.8779430, -4.7254086, 31.7561359, -33.0022659, 33.2213249
3: -8.8838043, 29.0562229, -8.9220181, 28.9051380, -32.3493500, 32.5662804
4: -9.9781418, 35.1261520, -10.0667772, 35.0420151, -43.2243958, 43.4130173
5: -11.0838261, 30.0737152, -11.1266308, 29.9172325, -38.0446396, 38.2963791
6: -38.7177429, 7.3866930, -38.6718979, 7.4385014, -44.4730148, 44.3292389
7: -15.1608362, 30.6771584, -15.2302608, 30.6082573, -41.5666275, 41.6924286
8: -15.3217840, 34.6293945, -15.4097891, 34.5378189, -46.8918457, 47.1123962
9: -10.2660618, 27.1823540, -10.2784357, 27.2028446, -35.7426910, 35.6556435
10: -28.5671272, 23.9550266, -28.4107018, 23.9641685, -50.7230072, 50.3484726
11: -35.9747696, 14.2903242, -35.7804337, 14.3045225, -49.6316910, 49.3537903
12: -49.6457367, 1.9074130, -49.3870621, 1.9830103, -44.2623672, 43.8962173
13: -28.7933083, 21.3916626, -28.7944260, 21.3332062, -49.5506592, 49.5798645
14: -71.0650635, -6.3936195, -70.9327927, -6.3960457, -64.6690216, 64.5391693
15: -17.3650169, 24.7466812, -17.3968391, 24.6821480, -42.0471649, 42.1435204
16: -27.4607639, 23.6316433, -27.4562454, 23.6788864, -48.4298248, 48.3079834
17: -71.2557755, -3.8272171, -71.1315765, -3.8767815, -67.3789978, 67.3043594
18: -34.8894463, 11.6569977, -34.7429657, 11.6580706, -40.7233620, 40.5674667
19: -25.7763443, 5.2644553, -25.6978474, 5.2653141, -29.9561234, 29.9150162
20: -26.5576363, 4.3724723, -26.4707870, 4.3722172, -29.1754761, 29.0570030
21: -31.4921570, 10.1506271, -31.3533745, 10.1532030, -40.5501938, 40.3118362
22: -33.6403351, 6.9475503, -33.6114845, 6.9554811, -38.4571686, 38.4396133
23: -26.9908047, 8.8758078, -26.9061203, 8.8804340, -35.2802505, 35.2200432
24: -23.2907524, 9.8523178, -23.2546444, 9.8374882, -32.8093529, 32.8242073
25: -29.2398605, 6.0510721, -29.2089729, 6.0704279, -34.3381729, 34.3346252
26: -43.1882057, 7.8349528, -42.9449387, 7.8300943, -43.9733963, 43.7024231
27: -26.6776505, 11.4537773, -26.6219940, 11.4713316, -37.7759552, 37.7195625
28: -29.6285763, 7.1222544, -29.5928478, 7.1382670, -36.5990143, 36.6015015
29: -32.6394730, 8.9703207, -32.5914574, 8.9927292, -41.6322021, 41.5617790
30: -37.6857910, 6.9678745, -37.5891075, 6.9665194, -44.6523094, 44.5569839
31: -31.4970512, 7.2611442, -31.4015827, 7.2594614, -37.6017151, 37.6730003
32: -33.7235527, 6.6922722, -33.6398621, 6.7302675, -40.4538193, 40.3321342
33: -43.9370728, 16.0426044, -43.9794464, 15.9827366, -57.0698624, 57.2460861
34: -50.6943741, -4.2784061, -50.7005196, -4.2962279, -42.1836166, 42.2866669
35: -40.9489555, 7.0664759, -40.9790535, 7.0302625, -43.7931900, 43.9333191
36: -44.4151840, 5.3433251, -44.4101486, 5.3849754, -45.4348221, 45.4124527
37: -59.4545784, 2.3388743, -59.4219131, 2.3467236, -55.1389771, 55.1776581
38: -50.8785934, 8.5546656, -50.8725510, 8.5827866, -59.4613800, 59.4272156
39: -52.1408730, 14.9303598, -52.1446457, 14.9017582, -67.0426331, 67.0750046
40: -47.8538094, 8.3289804, -47.8303108, 8.3094759, -53.1495056, 53.1278610
41: -31.8745575, 15.2112446, -31.8291416, 15.2355919, -45.5081177, 45.4229660
42: -27.2247982, 10.1238041, -27.1491108, 10.1577473, -36.8847580, 36.7273254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7391715, upper bound: 14.7711210
time: 41.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7711210
time: 40.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.9850636, 33.0339622, -23.9388123, 32.8723564, -54.3896103, 54.5203247
1: -7.7082024, 32.3907013, -7.6925325, 32.3150330, -36.5360794, 36.5537186
2: -4.8054395, 31.9155540, -4.7839403, 31.7594357, -33.1246948, 33.3224487
3: -8.9838896, 29.0902786, -8.9655132, 28.9095306, -32.4312668, 32.6519318
4: -10.1559992, 35.1748428, -10.1441383, 35.0454102, -43.3925552, 43.5408936
5: -11.1990395, 30.1193905, -11.1768999, 29.9228783, -38.1480179, 38.4007721
6: -38.7617340, 7.4871902, -38.6785583, 7.4815235, -44.5669403, 44.4258347
7: -15.3474064, 30.7255096, -15.3119411, 30.6105194, -41.7225647, 41.8346405
8: -15.5005455, 34.6848755, -15.4871931, 34.5433998, -47.0648880, 47.2529907
9: -10.4059963, 27.2273827, -10.3394003, 27.2068748, -35.8662643, 35.7660141
10: -28.6723557, 24.0117855, -28.4562054, 23.9735413, -50.8290634, 50.4515686
11: -36.0090256, 14.3307076, -35.7941856, 14.3151417, -49.6768112, 49.4130402
12: -49.6979523, 2.0812469, -49.3919449, 2.0567765, -44.4040527, 44.0213623
13: -28.8314457, 21.4245071, -28.8101425, 21.3454914, -49.6013565, 49.6514359
14: -71.2046432, -6.3391037, -70.9946213, -6.3880730, -64.8165741, 64.6555176
15: -17.4395294, 24.7787552, -17.4275017, 24.6872482, -42.1267776, 42.2062569
16: -27.6183739, 23.6825943, -27.5248070, 23.6841679, -48.5749512, 48.4313049
17: -71.3053131, -3.7929497, -71.1522369, -3.8701820, -67.4351349, 67.3592834
18: -34.9145432, 11.6875486, -34.7508545, 11.6702003, -40.7505035, 40.6188126
19: -25.8104858, 5.2784829, -25.7082672, 5.2711229, -30.0063705, 29.9400558
20: -26.5832500, 4.3888440, -26.4776325, 4.3798838, -29.2411156, 29.0739021
21: -31.5365753, 10.1694412, -31.3683186, 10.1605873, -40.6142273, 40.3386765
22: -33.6891937, 7.0072742, -33.6231918, 6.9813132, -38.5338440, 38.4994812
23: -27.0201263, 8.9122086, -26.9139099, 8.8955135, -35.3365936, 35.2669983
24: -23.3159809, 9.8702021, -23.2638035, 9.8465872, -32.8476868, 32.8516617
25: -29.2820301, 6.1287575, -29.2180290, 6.1044908, -34.4165802, 34.4185791
26: -43.2331161, 7.8826699, -42.9544067, 7.8508153, -44.0401382, 43.7455063
27: -26.7088642, 11.4906282, -26.6326141, 11.4822845, -37.8227997, 37.7718773
28: -29.6659374, 7.1909027, -29.6003952, 7.1681600, -36.6681671, 36.6782532
29: -32.6802444, 9.0303926, -32.6020813, 9.0189209, -41.6991653, 41.6324730
30: -37.7086830, 7.0046844, -37.5978394, 6.9802265, -44.6889114, 44.6025238
31: -31.5425606, 7.2917824, -31.4131298, 7.2718892, -37.6900330, 37.7154427
32: -33.7663651, 6.7830963, -33.6471062, 6.7704754, -40.5368423, 40.4302025
33: -44.0140457, 16.1326866, -43.9970093, 16.0225487, -57.1882248, 57.3385773
34: -50.7453041, -4.1667013, -50.7072182, -4.2473459, -42.2975388, 42.3658905
35: -41.0081253, 7.1772003, -40.9885826, 7.0789466, -43.9139404, 44.0131683
36: -44.4708214, 5.4691610, -44.4174423, 5.4403520, -45.5545120, 45.5193558
37: -59.5227280, 2.4055238, -59.4373741, 2.3756237, -55.2468262, 55.2079239
38: -50.9534416, 8.6596594, -50.8837051, 8.6280270, -59.5814667, 59.5433655
39: -52.2189445, 14.9540577, -52.1654968, 14.9114304, -67.1303711, 67.1195526
40: -47.9134636, 8.3620853, -47.8442879, 8.3232040, -53.2211456, 53.1700363
41: -31.9235039, 15.2860918, -31.8396740, 15.2675991, -45.5887756, 45.5004883
42: -27.2610130, 10.2123451, -27.1549454, 10.1955738, -36.9774017, 36.8116150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7840022, upper bound: 14.7711210
time: 35.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7840026, upper bound: 14.7711210
time: 34.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.7733364, 32.9825554, -23.9192657, 32.8990936, -54.2107010, 54.4466019
1: -7.5697956, 32.3565521, -7.6826248, 32.3402100, -36.4289169, 36.5063782
2: -4.6744642, 31.8938065, -4.8180466, 31.8048801, -33.0504150, 33.3297653
3: -8.8856859, 29.0586052, -8.9700890, 28.9215298, -32.3652115, 32.6197739
4: -9.9797592, 35.1414833, -10.1588659, 35.0920105, -43.2749557, 43.5192871
5: -11.0857277, 30.0779152, -11.1727734, 29.9396896, -38.0669708, 38.3557739
6: -38.7453461, 7.3881111, -38.7597313, 7.5725861, -44.6341476, 44.4134827
7: -15.1624355, 30.6883545, -15.2906361, 30.6431007, -41.6020813, 41.7644196
8: -15.3237486, 34.6446037, -15.4918976, 34.5877876, -46.9405670, 47.2100296
9: -10.2634783, 27.1842632, -10.3073015, 27.2368698, -35.7767029, 35.6911087
10: -28.5692387, 23.9664154, -28.4899158, 24.0159760, -50.7766190, 50.4798355
11: -35.9761505, 14.2998266, -35.8280792, 14.3442659, -49.6738739, 49.4099350
12: -49.6556053, 1.9092512, -49.4255333, 2.0268722, -44.3229675, 43.9402237
13: -28.7962971, 21.3920479, -28.8358898, 21.3607750, -49.5782318, 49.6316833
14: -71.0671387, -6.3682995, -71.0739517, -6.3206520, -64.7464905, 64.7056503
15: -17.3664436, 24.7488098, -17.4305458, 24.6988029, -42.0652466, 42.1793556
16: -27.4619827, 23.6347370, -27.5138340, 23.7053432, -48.4586334, 48.3699112
17: -71.2606125, -3.7871246, -71.3225021, -3.7636795, -67.4969330, 67.5353775
18: -34.8910599, 11.6554594, -34.7858543, 11.6818008, -40.7538948, 40.6063156
19: -25.7778854, 5.2675915, -25.7313309, 5.2884197, -29.9794235, 29.9458046
20: -26.5577850, 4.3736973, -26.4812698, 4.3952994, -29.1973076, 29.0720444
21: -31.4929829, 10.1533661, -31.3836956, 10.1765881, -40.5741119, 40.3396530
22: -33.6422806, 6.9503627, -33.6465034, 6.9806209, -38.4829865, 38.4970703
23: -26.9920502, 8.8781147, -26.9326668, 8.9046993, -35.3066330, 35.2480087
24: -23.2918968, 9.8579845, -23.3113861, 9.8736610, -32.8454170, 32.8847885
25: -29.2401962, 6.0558562, -29.2578926, 6.1052365, -34.3750916, 34.3945084
26: -43.1898727, 7.8343115, -42.9707680, 7.8431988, -43.9941254, 43.7335434
27: -26.6863842, 11.4546976, -26.6681824, 11.5352554, -37.8690872, 37.7699585
28: -29.6293297, 7.1244006, -29.6114140, 7.1807742, -36.6475449, 36.6223831
29: -32.6405029, 8.9732504, -32.6226349, 9.0175591, -41.6580620, 41.5958862
30: -37.6857185, 6.9705725, -37.6036148, 6.9951487, -44.6808662, 44.5741882
31: -31.4993877, 7.2710714, -31.4402504, 7.2998781, -37.6384354, 37.7125168
32: -33.7508087, 6.6930447, -33.7301559, 6.8671350, -40.6179428, 40.4232025
33: -43.9396820, 16.0439396, -44.0035667, 16.0264721, -57.1305389, 57.2789993
34: -50.7113342, -4.2775879, -50.7623749, -4.2055783, -42.2997360, 42.3493347
35: -40.9575348, 7.0671854, -41.0164490, 7.0779314, -43.8567963, 43.9692841
36: -44.4428978, 5.3441072, -44.4979935, 5.5078831, -45.5856323, 45.5002518
37: -59.4655647, 2.3405228, -59.4689064, 2.4298072, -55.2435379, 55.2271118
38: -50.8993721, 8.5560293, -50.9535370, 8.6876717, -59.5870438, 59.5095673
39: -52.1485825, 14.9313278, -52.1890259, 14.9498444, -67.0984268, 67.1203537
40: -47.8757591, 8.3303232, -47.9023781, 8.4278803, -53.2899475, 53.1994629
41: -31.8996620, 15.2123365, -31.9153690, 15.3688345, -45.6675568, 45.5082474
42: -27.2424469, 10.1252108, -27.2116241, 10.2580252, -37.0025940, 36.7879715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7682092, upper bound: 14.7711210
time: 37.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7262839, upper bound: 14.7711210
time: 40.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.9872398, 33.0412292, -24.0123787, 32.9024277, -54.4229660, 54.6010513
1: -7.7091999, 32.3987541, -7.7437391, 32.3427353, -36.5610580, 36.6145744
2: -4.8073125, 31.9314690, -4.8765545, 31.8081532, -33.1727982, 33.4308777
3: -8.9857750, 29.0926170, -9.0135727, 28.9258900, -32.4471436, 32.7054062
4: -10.1576366, 35.1901703, -10.2362127, 35.0953293, -43.4431305, 43.6471710
5: -11.2009106, 30.1236076, -11.2230492, 29.9453602, -38.1703491, 38.4601822
6: -38.7892799, 7.4885540, -38.7663460, 7.6156025, -44.7281113, 44.5100632
7: -15.3489923, 30.7367096, -15.3722763, 30.6454277, -41.7579346, 41.9066315
8: -15.5025082, 34.7000961, -15.5692348, 34.5933990, -47.1136169, 47.3505402
9: -10.4034262, 27.2293091, -10.3682041, 27.2409000, -35.9003067, 35.8015099
10: -28.6744499, 24.0232239, -28.5353947, 24.0253410, -50.8826752, 50.5829391
11: -36.0104294, 14.3402338, -35.8418350, 14.3549557, -49.7189636, 49.4691620
12: -49.7079353, 2.0830655, -49.4304199, 2.1006413, -44.4646683, 44.0653610
13: -28.8344154, 21.4249439, -28.8516026, 21.3730469, -49.6289673, 49.7032852
14: -71.2066956, -6.3137798, -71.1357574, -6.3127365, -64.8939590, 64.8219757
15: -17.4409485, 24.7809143, -17.4612465, 24.7039127, -42.1448593, 42.2421608
16: -27.6195297, 23.6857433, -27.5824356, 23.7106304, -48.6036987, 48.4932022
17: -71.3102036, -3.7528763, -71.3431091, -3.7570801, -67.5531235, 67.5902328
18: -34.9161301, 11.6860304, -34.7937737, 11.6938906, -40.7810211, 40.6577263
19: -25.8120632, 5.2816234, -25.7417526, 5.2941995, -30.0296707, 29.9708786
20: -26.5834026, 4.3900623, -26.4881077, 4.4029870, -29.2629700, 29.0889511
21: -31.5373993, 10.1721525, -31.3987103, 10.1839809, -40.6381378, 40.3665085
22: -33.6911583, 7.0100560, -33.6582794, 7.0064559, -38.5597610, 38.5570145
23: -27.0213470, 8.9144897, -26.9404659, 8.9197922, -35.3630371, 35.2949829
24: -23.3170967, 9.8758707, -23.3205357, 9.8827333, -32.8837738, 32.9122543
25: -29.2824173, 6.1335416, -29.2669449, 6.1393013, -34.4535065, 34.4784698
26: -43.2347984, 7.8819995, -42.9801941, 7.8639207, -44.0608826, 43.7766571
27: -26.7176228, 11.4915352, -26.6787720, 11.5462780, -37.9159470, 37.8223228
28: -29.6666756, 7.1930466, -29.6189766, 7.2106533, -36.7167664, 36.6991348
29: -32.6812706, 9.0332994, -32.6332550, 9.0437069, -41.7249756, 41.6665535
30: -37.7085762, 7.0074291, -37.6123734, 7.0088320, -44.7174072, 44.6198044
31: -31.5449162, 7.3017015, -31.4518070, 7.3122950, -37.7267113, 37.7549438
32: -33.7936325, 6.7838745, -33.7373810, 6.9073887, -40.7010193, 40.5212555
33: -44.0166397, 16.1340828, -44.0212097, 16.0663681, -57.2490082, 57.3714447
34: -50.7622604, -4.1659231, -50.7690392, -4.1566954, -42.4136887, 42.4285278
35: -41.0167427, 7.1778955, -41.0259895, 7.1266255, -43.9775620, 44.0491257
36: -44.4985619, 5.4699907, -44.5053101, 5.5632877, -45.7053528, 45.6070480
37: -59.5337029, 2.4071927, -59.4843445, 2.4587369, -55.3515320, 55.2574615
38: -50.9742050, 8.6610889, -50.9646568, 8.7329445, -59.7071495, 59.6257477
39: -52.2266464, 14.9550428, -52.2098351, 14.9595919, -67.1862411, 67.1648788
40: -47.9354477, 8.3634262, -47.9164314, 8.4416294, -53.3616638, 53.2416458
41: -31.9485893, 15.2871923, -31.9258842, 15.4008455, -45.7482071, 45.5857391
42: -27.2786674, 10.2137432, -27.2174740, 10.2958565, -37.0952148, 36.8722534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=178, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8130228, upper bound: 14.7711210
time: 39.30 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8130232, upper bound: 14.7711210
time: 41.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 82.87 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6951250, upper bound: 14.7919458
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6951250, upper bound: 14.8130237
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7399635, upper bound: 14.7919458
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7399635, upper bound: 14.8130237
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6972172, upper bound: 14.7898460
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6972172, upper bound: 14.8130237
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7420616, upper bound: 14.7898460
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7420616, upper bound: 14.8130237
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7262839, upper bound: 14.7898460
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7262839, upper bound: 14.8130237
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7711206, upper bound: 14.7898460
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7711206, upper bound: 14.8130237
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7391715, upper bound: 14.7711210
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7711210
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7840022, upper bound: 14.7711210
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7840026, upper bound: 14.7711210
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7682092, upper bound: 14.7711210
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.7262839, upper bound: 14.7711210
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.8130228, upper bound: 14.7711210
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.87
Output dim: 2, lower bound: -14.8130232, upper bound: 14.7711210

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.5435085, 32.7309723, -23.8395004, 32.9978218, -54.1162033, 54.8715820
1: -7.4647827, 32.2671471, -7.6216364, 32.3697014, -36.1125031, 36.3710823
2: -4.4838681, 31.6635532, -4.6907659, 31.8853912, -32.9144135, 32.9616089
3: -8.6969795, 28.7972107, -8.8925934, 29.0701714, -32.0153275, 32.3024979
4: -9.7928839, 34.9393349, -10.0211515, 35.1411858, -43.1795120, 43.3372574
5: -10.8459625, 29.7682686, -11.0802364, 30.0892715, -38.0518951, 38.1065903
6: -38.5739708, 7.3252058, -38.6886482, 7.4393024, -43.4678040, 44.2795258
7: -14.9855566, 30.5463600, -15.2312775, 30.6987247, -40.3103027, 41.5535431
8: -15.1264277, 34.3991890, -15.3555584, 34.6454849, -47.1715393, 46.8202057
9: -10.1158953, 27.0009460, -10.3340044, 27.1588707, -35.5462418, 35.9941902
10: -28.1219826, 23.5706024, -28.6112633, 23.8437119, -50.1036453, 50.3412094
11: -35.4800797, 14.0074425, -35.9660034, 14.1983862, -49.5256958, 49.3339539
12: -49.1530075, 1.4623423, -49.6668625, 1.8608880, -43.5264969, 43.8091278
13: -28.7354946, 21.1599503, -28.8027382, 21.3672581, -49.5135040, 49.9626884
14: -70.7025146, -6.6836662, -71.1271744, -6.4864979, -64.2160187, 64.4435120
15: -17.2311420, 24.5264130, -17.3626518, 24.7470741, -41.9782181, 41.8890648
16: -27.1850262, 23.4690819, -27.5271378, 23.6034641, -47.5637665, 48.2817001
17: -71.0323868, -4.1184120, -71.2632599, -3.9530983, -67.0792847, 67.1448517
18: -34.6373291, 11.6065302, -34.8849564, 11.6596699, -40.4462776, 40.4654808
19: -25.5759010, 5.1869898, -25.7783108, 5.2368073, -29.7820969, 29.7104073
20: -26.3381863, 4.2457690, -26.5575714, 4.3363543, -29.3137398, 29.0942383
21: -31.1554012, 9.9771271, -31.5003052, 10.0924683, -40.2574692, 40.3947601
22: -33.5077095, 6.8334746, -33.6636391, 6.9493966, -38.3108292, 36.7957840
23: -26.7825737, 8.7811632, -26.9977150, 8.8604851, -35.1214905, 34.9187241
24: -23.1397800, 9.8189802, -23.2777824, 9.8433456, -32.6654015, 32.4962654
25: -29.1067467, 5.9445391, -29.2554398, 6.0508356, -34.2392044, 32.8582840
26: -42.7778702, 7.5555172, -43.1984482, 7.7874522, -43.7237244, 43.7334671
27: -26.5097904, 11.4227467, -26.6524792, 11.4736538, -37.9834442, 37.7059784
28: -29.5188370, 7.0632467, -29.6377964, 7.1491694, -36.5165787, 36.4000244
29: -32.5128632, 8.8555470, -32.6538620, 8.9662704, -41.4791336, 41.5094070
30: -37.4768791, 6.8268719, -37.6817398, 6.9439926, -44.4208717, 44.5086136
31: -31.2397194, 7.2062116, -31.5047970, 7.2439590, -37.4318657, 37.1722870
32: -33.5294113, 6.5498037, -33.7026367, 6.6924391, -39.9320755, 40.2524414
33: -43.7108192, 15.8172817, -43.9163094, 16.0739212, -57.0012054, 56.9941254
34: -50.5612221, -4.4343901, -50.6656265, -4.2249174, -42.1619644, 42.0680771
35: -40.7730103, 6.8669853, -40.9225502, 7.1181779, -43.7521439, 43.7745361
36: -44.3465881, 5.2529173, -44.4013672, 5.3989635, -45.3679657, 45.3890762
37: -59.2907982, 2.2604852, -59.4561234, 2.3563280, -54.9836502, 55.4566650
38: -50.7408218, 8.4416866, -50.8723907, 8.5870829, -59.3279037, 59.3140793
39: -52.0045586, 14.7939482, -52.1543999, 14.9154406, -66.9199982, 66.9483490
40: -47.6948700, 8.2524614, -47.8267441, 8.3382444, -52.9462051, 53.0494308
41: -31.7025928, 15.1356020, -31.8449554, 15.2296953, -44.9554672, 45.3721008
42: -27.0134773, 9.9494762, -27.2115669, 10.1104965, -36.5001221, 36.6094513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6281871, upper bound: 14.8130234
time: 27.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 39.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.7574425, 32.7889061, -23.9326344, 33.0011139, -54.3286285, 54.2605820
1: -7.6040649, 32.3092766, -7.6827507, 32.3722267, -36.4686127, 36.4792900
2: -4.6166458, 31.7011967, -4.7492647, 31.8887081, -33.1316071, 33.0627594
3: -8.7968626, 28.8311729, -8.9360199, 29.0745735, -32.4787521, 32.3880539
4: -9.9708614, 34.9879456, -10.0984783, 35.1445427, -43.3477631, 43.3252411
5: -10.9611607, 29.8137398, -11.1304798, 30.0949287, -38.1552887, 38.0467606
6: -38.6171989, 7.4258060, -38.6952972, 7.4823189, -44.2794189, 44.3761673
7: -15.1715431, 30.5946579, -15.3129387, 30.7009869, -41.6659851, 41.6957169
8: -15.3047028, 34.4547310, -15.4329691, 34.6510773, -47.0449219, 46.9607468
9: -10.2559280, 27.0454082, -10.3949385, 27.1628914, -35.6699219, 35.7194252
10: -28.2271576, 23.6272202, -28.6567154, 23.8530884, -50.2559891, 50.4441147
11: -35.5131531, 14.0477657, -35.9798050, 14.2090836, -49.0670700, 49.3931198
12: -49.2050705, 1.6363997, -49.6717682, 1.9346404, -43.7807388, 43.9344864
13: -28.7735443, 21.1921196, -28.8184814, 21.3795090, -49.5641632, 49.4258423
14: -70.8421173, -6.6292629, -71.1889343, -6.4785175, -64.3636017, 64.5596695
15: -17.3058147, 24.5577812, -17.3932991, 24.7521915, -42.0580063, 41.9510803
16: -27.3423004, 23.5199280, -27.5957108, 23.6087761, -48.2669525, 48.4048767
17: -71.0819550, -4.0848503, -71.2839050, -3.9464760, -67.1354828, 67.1990509
18: -34.6623383, 11.6367760, -34.8928947, 11.6718140, -40.4734077, 40.6879044
19: -25.6100292, 5.2007856, -25.7887039, 5.2425947, -29.8322754, 29.9445839
20: -26.3632908, 4.2621269, -26.5644207, 4.3440361, -29.0030022, 29.1111679
21: -31.1996136, 9.9958000, -31.5152702, 10.0998878, -40.1911621, 40.4215164
22: -33.5565567, 6.8925295, -33.6753731, 6.9752231, -38.3874969, 38.4486694
23: -26.8118362, 8.8174114, -27.0054817, 8.8755684, -35.1777878, 35.2796631
24: -23.1649399, 9.8367500, -23.2868958, 9.8524761, -32.7036438, 32.7880554
25: -29.1488972, 6.0216289, -29.2645016, 6.0849233, -34.3175583, 34.3331223
26: -42.8224716, 7.6032643, -43.2078476, 7.8082027, -43.5637741, 43.7765656
27: -26.5404987, 11.4595413, -26.6630630, 11.4846563, -37.6482697, 37.7584381
28: -29.5560970, 7.1318240, -29.6453400, 7.1790538, -36.5856400, 36.6061020
29: -32.5536270, 8.9151516, -32.6644592, 8.9924097, -41.5460358, 41.5796127
30: -37.4989929, 6.8635035, -37.6904449, 6.9577103, -44.4567032, 44.5539474
31: -31.2852459, 7.2365184, -31.5163059, 7.2563972, -37.5200233, 37.6500702
32: -33.5718193, 6.6406879, -33.7098656, 6.7326956, -40.3045158, 40.3505554
33: -43.7876434, 15.9067326, -43.9339218, 16.1137810, -57.1194305, 57.0377197
34: -50.6120529, -4.3227129, -50.6723328, -4.1760263, -42.2757950, 42.1802444
35: -40.8321571, 6.9773088, -40.9320526, 7.1668692, -43.8728409, 43.7598572
36: -44.4021950, 5.3787289, -44.4086723, 5.4543524, -45.4876633, 45.4106903
37: -59.3588486, 2.3267679, -59.4715843, 2.3852692, -55.0913544, 55.1033859
38: -50.8154564, 8.5468102, -50.8834915, 8.6323366, -59.4477921, 59.4303017
39: -52.0825005, 14.8172007, -52.1751900, 14.9251375, -67.0076370, 66.9923935
40: -47.7536659, 8.2855988, -47.8407516, 8.3519459, -53.0914230, 53.0916214
41: -31.7511139, 15.2105227, -31.8555298, 15.2616835, -45.3834000, 45.4496613
42: -27.0493088, 10.0381107, -27.2173481, 10.1483002, -36.5943985, 36.6938057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6281871, upper bound: 14.8130238
time: 48.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 40.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -23.5456924, 32.7382240, -23.9131699, 33.0278664, -54.1496429, 54.9508820
1: -7.4657373, 32.2751694, -7.6728554, 32.3973999, -36.1374817, 36.4318924
2: -4.4857020, 31.6794395, -4.7834396, 31.9341297, -32.9627380, 33.0701294
3: -8.6988707, 28.7995491, -8.9405823, 29.0865631, -32.0305481, 32.3558807
4: -9.7945127, 34.9546814, -10.1131601, 35.1910858, -43.2299805, 43.4433670
5: -10.8478689, 29.7724915, -11.1263990, 30.1116829, -38.0742340, 38.1640930
6: -38.6015282, 7.3265877, -38.7765083, 7.5734005, -43.6287537, 44.3637543
7: -14.9870396, 30.5575581, -15.2916679, 30.7335815, -40.3451157, 41.6254730
8: -15.1283884, 34.4144592, -15.4376154, 34.6953506, -47.2222366, 46.9177933
9: -10.1133184, 27.0028648, -10.3628368, 27.1928864, -35.5802917, 36.0281258
10: -28.1240921, 23.5820770, -28.6904831, 23.8955173, -50.1569519, 50.4725876
11: -35.4814377, 14.0169621, -36.0136337, 14.2382631, -49.5642319, 49.3900452
12: -49.1629295, 1.4642167, -49.7053146, 1.9048176, -43.5834503, 43.8530731
13: -28.7384682, 21.1603088, -28.8441696, 21.3946896, -49.5410233, 50.0044785
14: -70.7045898, -6.6583672, -71.2684174, -6.4110737, -64.2935181, 64.6100464
15: -17.2325916, 24.5285587, -17.3963394, 24.7637119, -41.9963036, 41.9248962
16: -27.1861515, 23.4721680, -27.5847874, 23.6299362, -47.5923309, 48.3435898
17: -71.0372620, -4.0783005, -71.4541626, -3.8399849, -67.1972809, 67.3758621
18: -34.6389427, 11.6049709, -34.9279518, 11.6834202, -40.4768677, 40.5029678
19: -25.5774593, 5.1900940, -25.8119087, 5.2599177, -29.8053818, 29.7401505
20: -26.3383198, 4.2470012, -26.5680504, 4.3594675, -29.3333588, 29.1092949
21: -31.1562214, 9.9798679, -31.5306664, 10.1158743, -40.2808228, 40.4225082
22: -33.5096550, 6.8362370, -33.6987610, 6.9745994, -38.3367157, 36.8357086
23: -26.7838326, 8.7834387, -27.0242233, 8.8848381, -35.1478882, 34.9470177
24: -23.1409149, 9.8246574, -23.3346367, 9.8796043, -32.7015228, 32.5553932
25: -29.1070995, 5.9493055, -29.3043823, 6.0856552, -34.2761383, 32.9123039
26: -42.7795792, 7.5548477, -43.2242393, 7.8005333, -43.7436371, 43.7646790
27: -26.5185261, 11.4236279, -26.6985435, 11.5376129, -38.0561371, 37.7562637
28: -29.5195694, 7.0654078, -29.6562519, 7.1916375, -36.5649796, 36.4211731
29: -32.5138550, 8.8584576, -32.6850433, 8.9910641, -41.5049210, 41.5435028
30: -37.4767647, 6.8295803, -37.6962128, 6.9726267, -44.4493904, 44.5257950
31: -31.2420769, 7.2161164, -31.5435219, 7.2844439, -37.4686203, 37.2105331
32: -33.5566788, 6.5505695, -33.7928848, 6.8293190, -40.0973282, 40.3434525
33: -43.7134018, 15.8185759, -43.9404449, 16.1177197, -57.0617447, 57.0233612
34: -50.5781708, -4.4336395, -50.7275162, -4.1342697, -42.2781448, 42.1309662
35: -40.7816429, 6.8676734, -40.9599380, 7.1658673, -43.8157883, 43.8089905
36: -44.3743324, 5.2536769, -44.4892387, 5.5218849, -45.5189056, 45.4768982
37: -59.3018036, 2.2621207, -59.5032120, 2.4394388, -55.0883102, 55.5044327
38: -50.7616806, 8.4430246, -50.9535408, 8.6920223, -59.4537048, 59.3965645
39: -52.0122375, 14.7948523, -52.1987381, 14.9636259, -66.9758606, 66.9935913
40: -47.7169151, 8.2537928, -47.8989258, 8.4566402, -53.0877686, 53.1210938
41: -31.7276630, 15.1367283, -31.9311275, 15.3629456, -45.1142578, 45.4573441
42: -27.0311546, 9.9509058, -27.2740650, 10.2108097, -36.6176300, 36.6701279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6572734, upper bound: 14.8130234
time: 31.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6572734, upper bound: 14.7752221
time: 32.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.7596073, 32.7961502, -24.0062809, 33.0311928, -54.3619690, 54.3413544
1: -7.6050005, 32.3173294, -7.7339735, 32.3998985, -36.4935760, 36.5401001
2: -4.6184998, 31.7170792, -4.8419566, 31.9374447, -33.1797180, 33.1712189
3: -8.7987556, 28.8335419, -8.9840813, 29.0909710, -32.4946136, 32.4414711
4: -9.9724874, 35.0032692, -10.1905499, 35.1944542, -43.3982697, 43.4315338
5: -10.9630547, 29.8179550, -11.1766243, 30.1173573, -38.1775665, 38.1062012
6: -38.6448174, 7.4271450, -38.7831306, 7.6164160, -44.4406281, 44.4604111
7: -15.1730843, 30.6058750, -15.3733234, 30.7358856, -41.7013474, 41.7676544
8: -15.3066988, 34.4699745, -15.5149784, 34.7009697, -47.0935974, 47.0583267
9: -10.2533388, 27.0473537, -10.4237261, 27.1968994, -35.7039795, 35.7549210
10: -28.2293301, 23.6386452, -28.7359238, 23.9049377, -50.3096161, 50.5755768
11: -35.5145493, 14.0573025, -36.0273438, 14.2488918, -49.1092758, 49.4492188
12: -49.2149734, 1.6382594, -49.7102051, 1.9786315, -43.8414764, 43.9784698
13: -28.7765007, 21.1925735, -28.8599434, 21.4069881, -49.5916672, 49.4776611
14: -70.8441620, -6.6039257, -71.3301849, -6.4031868, -64.4409790, 64.7262573
15: -17.3072128, 24.5599003, -17.4270229, 24.7688370, -42.0760498, 41.9869232
16: -27.3434486, 23.5230179, -27.6533642, 23.6352329, -48.2955627, 48.4667816
17: -71.0868683, -4.0447388, -71.4748917, -3.8334236, -67.2534485, 67.4301529
18: -34.6639633, 11.6352539, -34.9358406, 11.6955614, -40.5039673, 40.7270355
19: -25.6116142, 5.2039404, -25.8223553, 5.2657142, -29.8555908, 29.9754715
20: -26.3634434, 4.2633567, -26.5748863, 4.3671923, -29.0248413, 29.1262436
21: -31.2004128, 9.9985008, -31.5456467, 10.1232986, -40.2150726, 40.4492874
22: -33.5584908, 6.8953581, -33.7105103, 7.0004430, -38.4134064, 38.5064392
23: -26.8131027, 8.8197069, -27.0320072, 8.8999023, -35.2042160, 35.3077431
24: -23.1660824, 9.8424149, -23.3437729, 9.8887215, -32.7397652, 32.8487244
25: -29.1492290, 6.0264306, -29.3134575, 6.1197138, -34.3545074, 34.3931084
26: -42.8241463, 7.6025896, -43.2336540, 7.8212872, -43.5844727, 43.8077469
27: -26.5492706, 11.4604483, -26.7091179, 11.5486546, -37.7414093, 37.8087311
28: -29.5568619, 7.1339707, -29.6638432, 7.2215004, -36.6341019, 36.6268387
29: -32.5546265, 8.9180756, -32.6956406, 9.0172110, -41.5718384, 41.6137161
30: -37.4989128, 6.8662548, -37.7049942, 6.9863300, -44.4852448, 44.5712509
31: -31.2876053, 7.2464347, -31.5550270, 7.2968454, -37.5567284, 37.6896820
32: -33.5990448, 6.6414871, -33.8001251, 6.8695679, -40.4686127, 40.4416122
33: -43.7902412, 15.9080238, -43.9581108, 16.1575794, -57.1800537, 57.0705109
34: -50.6290131, -4.3219166, -50.7341957, -4.0853653, -42.3919907, 42.2428589
35: -40.8407707, 6.9780254, -40.9694214, 7.2145243, -43.9365234, 43.7958145
36: -44.4299240, 5.3794384, -44.4964867, 5.5772963, -45.6385193, 45.4983368
37: -59.3698578, 2.3283682, -59.5187035, 2.4683599, -55.1960526, 55.1528854
38: -50.8362961, 8.5481567, -50.9646873, 8.7372227, -59.5735168, 59.5128441
39: -52.0902176, 14.8181429, -52.2195244, 14.9733477, -67.0635681, 67.0376663
40: -47.7756577, 8.2869244, -47.9129181, 8.4703884, -53.2319489, 53.1633072
41: -31.7761955, 15.2116060, -31.9416428, 15.3949594, -45.5429306, 45.5348358
42: -27.0669956, 10.0395098, -27.2798328, 10.2486038, -36.7121887, 36.7544556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7021407, upper bound: 14.7752221
time: 66.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7021407, upper bound: 14.7752221
time: 35.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.6287079, 32.8806686, -23.8414707, 32.8674202, -54.0645676, 54.2620773
1: -7.4981365, 32.3271713, -7.6288757, 32.3120079, -36.3124237, 36.4178734
2: -4.5446854, 31.7924213, -4.7223454, 31.7552185, -32.9321976, 33.1309509
3: -8.7680941, 28.9737396, -8.9150143, 28.9036255, -32.2957306, 32.4884224
4: -9.8395433, 35.0467491, -10.0598383, 35.0402489, -43.1220627, 43.3255997
5: -10.9262161, 29.9576530, -11.1231174, 29.9152622, -37.9520340, 38.1735001
6: -38.6270256, 7.3437405, -38.6686935, 7.4238081, -44.2894592, 44.2408905
7: -15.0579157, 30.6457500, -15.2261848, 30.6065350, -41.4800415, 41.6547394
8: -15.1745453, 34.5156364, -15.4054012, 34.5361633, -46.8002167, 46.9947815
9: -10.1784630, 27.0380440, -10.2770548, 27.1975765, -35.6448364, 35.5972519
10: -28.3359241, 23.6558323, -28.4092960, 23.9526024, -50.4731445, 50.2174301
11: -35.7018204, 14.0601902, -35.7773743, 14.2950993, -49.3515472, 49.1846085
12: -49.4326706, 1.5841122, -49.3846550, 1.9718251, -44.0396271, 43.6428757
13: -28.7580986, 21.2621117, -28.7921715, 21.3285942, -49.5032501, 49.4434586
14: -70.9055023, -6.6137428, -70.9302216, -6.4020348, -64.5034637, 64.3164825
15: -17.2486725, 24.6091690, -17.3878212, 24.6799507, -41.9286232, 41.9969902
16: -27.2799606, 23.4795761, -27.4528999, 23.6699619, -48.2565613, 48.2387772
17: -71.1796570, -4.0391884, -71.1292572, -3.8855343, -67.2941208, 67.0900726
18: -34.8001785, 11.6374664, -34.7408524, 11.6568460, -40.6118126, 40.5146713
19: -25.6797810, 5.2061310, -25.6959095, 5.2623386, -29.8767242, 29.8577690
20: -26.4475689, 4.2766814, -26.4692287, 4.3682632, -29.0762253, 29.0229378
21: -31.3293533, 10.0198460, -31.3509617, 10.1481781, -40.3735962, 40.2792435
22: -33.5844803, 6.8765483, -33.6019096, 6.9525671, -38.3997803, 38.3638535
23: -26.8898849, 8.8061543, -26.9044743, 8.8777313, -35.2065887, 35.1781464
24: -23.2041073, 9.8342352, -23.2500954, 9.8363991, -32.7046700, 32.7673264
25: -29.1736069, 5.9761477, -29.2061577, 6.0679278, -34.2804337, 34.2418137
26: -43.0537758, 7.6626081, -42.9416771, 7.8254757, -43.8196945, 43.5791473
27: -26.5867958, 11.4398479, -26.6147289, 11.4701366, -37.6750336, 37.6968231
28: -29.5842476, 7.0887160, -29.5912018, 7.1357708, -36.5451202, 36.5288620
29: -32.5876236, 8.8917561, -32.5898819, 8.9889593, -41.5765839, 41.4816360
30: -37.5872498, 6.8658457, -37.5866470, 6.9616537, -44.5489044, 44.4524918
31: -31.3724537, 7.2215271, -31.3994846, 7.2573318, -37.5014877, 37.5507965
32: -33.6086655, 6.5784817, -33.6380501, 6.7248058, -40.3334732, 40.2165298
33: -43.7643204, 15.9296017, -43.9719238, 15.9811325, -56.9477081, 57.1267319
34: -50.5928764, -4.3527746, -50.6960754, -4.2972445, -42.1170349, 42.2113800
35: -40.8096695, 6.9672432, -40.9729080, 7.0293751, -43.7004089, 43.8297119
36: -44.3632202, 5.2905588, -44.4057693, 5.3821745, -45.3741150, 45.3495712
37: -59.3651390, 2.2914901, -59.4178619, 2.3453298, -55.0270996, 55.0939331
38: -50.7859688, 8.4805746, -50.8677635, 8.5778008, -59.3637695, 59.3483391
39: -52.0550156, 14.8439608, -52.1400452, 14.8987398, -66.9537582, 66.9840088
40: -47.7463112, 8.2914848, -47.8265190, 8.3069210, -53.0408325, 53.0859146
41: -31.7547150, 15.1568918, -31.8262482, 15.2333994, -45.3782120, 45.3682327
42: -27.0889454, 9.9733782, -27.1466160, 10.1500168, -36.6973648, 36.5455360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6593509, upper bound: 14.8130234
time: 38.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6593509, upper bound: 14.7752221
time: 53.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.8426208, 32.9393082, -23.9346027, 32.8707428, -54.2769318, 54.4164124
1: -7.6375523, 32.3693619, -7.6899796, 32.3144989, -36.4445953, 36.5260544
2: -4.6775455, 31.8300381, -4.7808938, 31.7585182, -33.0546494, 33.2320862
3: -8.8681402, 29.0077629, -8.9584618, 28.9079895, -32.3776550, 32.5740318
4: -10.0174809, 35.0954590, -10.1371832, 35.0436554, -43.2903137, 43.4535065
5: -11.0413818, 30.0032272, -11.1733990, 29.9209480, -38.0553589, 38.2778015
6: -38.6709442, 7.4442644, -38.6753273, 7.4668474, -44.3834229, 44.3375092
7: -15.2444630, 30.6940365, -15.3078375, 30.6088581, -41.6359406, 41.7969666
8: -15.3533669, 34.5711517, -15.4827576, 34.5417290, -46.9733810, 47.1352921
9: -10.3184299, 27.0830097, -10.3380146, 27.2015953, -35.7684326, 35.7075500
10: -28.4412098, 23.7125549, -28.4547558, 23.9619274, -50.5792313, 50.3204269
11: -35.7360458, 14.1004963, -35.7912102, 14.3057976, -49.3966064, 49.2437897
12: -49.4849472, 1.7580810, -49.3894768, 2.0456243, -44.1813202, 43.7681732
13: -28.7961655, 21.2949390, -28.8078575, 21.3408623, -49.5539322, 49.5149612
14: -71.0451126, -6.5592766, -70.9920273, -6.3941288, -64.6509857, 64.4327545
15: -17.3232574, 24.6412506, -17.4184875, 24.6850357, -42.0082932, 42.0597382
16: -27.4375839, 23.5305271, -27.5214787, 23.6752014, -48.4016113, 48.3619995
17: -71.2292099, -4.0050430, -71.1498795, -3.8789845, -67.3502274, 67.1448364
18: -34.8251953, 11.6679611, -34.7487411, 11.6689978, -40.6389313, 40.5661011
19: -25.7139435, 5.2201042, -25.7063084, 5.2681661, -29.9269638, 29.8828812
20: -26.4731464, 4.2930322, -26.4760742, 4.3759437, -29.1417999, 29.0398369
21: -31.3737392, 10.0386190, -31.3658981, 10.1556091, -40.4376068, 40.3060684
22: -33.6333313, 6.9363070, -33.6136131, 6.9784117, -38.4764252, 38.4237366
23: -26.9191704, 8.8425694, -26.9122391, 8.8928356, -35.2629623, 35.2251434
24: -23.2292747, 9.8521595, -23.2592354, 9.8454895, -32.7429619, 32.7947769
25: -29.2157841, 6.0538559, -29.2152367, 6.1020117, -34.3588028, 34.3258629
26: -43.0985947, 7.7103033, -42.9511375, 7.8462219, -43.8862762, 43.6222534
27: -26.6179390, 11.4766865, -26.6253452, 11.4811497, -37.7218323, 37.7491493
28: -29.6215763, 7.1573772, -29.5987473, 7.1656737, -36.6142197, 36.6056595
29: -32.6283493, 8.9517994, -32.6004486, 9.0151033, -41.6434517, 41.5522461
30: -37.6101074, 6.9025917, -37.5953827, 6.9753551, -44.5854645, 44.4979744
31: -31.4179382, 7.2521620, -31.4110508, 7.2698016, -37.5896988, 37.5932388
32: -33.6515274, 6.6693544, -33.6452827, 6.7650709, -40.4165993, 40.3146362
33: -43.8412170, 16.0198917, -43.9895248, 16.0209618, -57.0660400, 57.2191696
34: -50.6437149, -4.2410765, -50.7027702, -4.2483325, -42.2309341, 42.2906494
35: -40.8688278, 7.0779548, -40.9824257, 7.0780373, -43.8211365, 43.9095764
36: -44.4188347, 5.4164100, -44.4130440, 5.4375672, -45.4937515, 45.4564056
37: -59.4332047, 2.3581915, -59.4332771, 2.3742452, -55.1349487, 55.1242065
38: -50.8607101, 8.5856571, -50.8789101, 8.6231060, -59.4838181, 59.4645691
39: -52.1329765, 14.8677073, -52.1608391, 14.9084568, -67.0414352, 67.0285492
40: -47.8058891, 8.3245993, -47.8405037, 8.3206625, -53.1122513, 53.1281509
41: -31.8036671, 15.2317495, -31.8368111, 15.2654238, -45.4588470, 45.4457550
42: -27.1251583, 10.0619411, -27.1524410, 10.1878500, -36.7900391, 36.6298943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7042232, upper bound: 14.8130234
time: 34.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7042232, upper bound: 14.7752221
time: 43.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -23.6308899, 32.8879471, -23.9150486, 32.8974838, -54.0980072, 54.3427277
1: -7.4990931, 32.3352203, -7.6800632, 32.3396606, -36.3373871, 36.4787369
2: -4.5465441, 31.8083172, -4.8150043, 31.8039703, -32.9803467, 33.2393646
3: -8.7700024, 28.9761372, -8.9630642, 28.9200172, -32.3115768, 32.5419121
4: -9.8411808, 35.0620956, -10.1519222, 35.0902405, -43.1725922, 43.4318771
5: -10.9281178, 29.9618378, -11.1692686, 29.9377174, -37.9743500, 38.2329025
6: -38.6546631, 7.3451033, -38.7565155, 7.5579090, -44.4505768, 44.3251648
7: -15.0594568, 30.6569767, -15.2865391, 30.6414394, -41.5154266, 41.7267456
8: -15.1765242, 34.5308647, -15.4874697, 34.5860786, -46.8489761, 47.0923767
9: -10.1758862, 27.0399742, -10.3059139, 27.2316341, -35.6788406, 35.6327858
10: -28.3380413, 23.6672401, -28.4884605, 24.0044098, -50.5266724, 50.3487930
11: -35.7031898, 14.0696583, -35.8250656, 14.3348684, -49.3936844, 49.2406845
12: -49.4425888, 1.5859361, -49.4230804, 2.0157156, -44.1003036, 43.6869049
13: -28.7611046, 21.2625427, -28.8335953, 21.3561306, -49.5308685, 49.4952545
14: -70.9075928, -6.5884094, -71.0713425, -6.3266354, -64.5809555, 64.4829330
15: -17.2500992, 24.6113358, -17.4215584, 24.6965771, -41.9466782, 42.0328941
16: -27.2811069, 23.4827023, -27.5105057, 23.6964340, -48.2853241, 48.3005981
17: -71.1845245, -3.9990845, -71.3201447, -3.7724380, -67.4120865, 67.3210602
18: -34.8018150, 11.6359291, -34.7837906, 11.6805506, -40.6423302, 40.5536232
19: -25.6813755, 5.2092438, -25.7293873, 5.2854686, -29.9000244, 29.8885918
20: -26.4477158, 4.2779260, -26.4796982, 4.3913431, -29.0980453, 29.0379791
21: -31.3301811, 10.0225563, -31.3813133, 10.1715784, -40.3975067, 40.3070679
22: -33.5864296, 6.8793187, -33.6369362, 6.9777007, -38.4256668, 38.4213562
23: -26.8911514, 8.8084383, -26.9310379, 8.9020119, -35.2329941, 35.2061310
24: -23.2052422, 9.8399258, -23.3068314, 9.8725977, -32.7408142, 32.8278961
25: -29.1739845, 5.9809475, -29.2550774, 6.1027470, -34.3173828, 34.3017235
26: -43.0554733, 7.6619630, -42.9674644, 7.8385963, -43.8403473, 43.6103287
27: -26.5955563, 11.4407501, -26.6609039, 11.5341244, -37.7682190, 37.7472229
28: -29.5849915, 7.0908499, -29.6097717, 7.1782308, -36.5936356, 36.5497055
29: -32.5886459, 8.8946743, -32.6210365, 9.0138111, -41.6024551, 41.5157089
30: -37.5871277, 6.8685837, -37.6011696, 6.9902697, -44.5773964, 44.4697533
31: -31.3748531, 7.2314572, -31.4381752, 7.2977967, -37.5382690, 37.5903320
32: -33.6359138, 6.5792599, -33.7282906, 6.8616810, -40.4975967, 40.3075485
33: -43.7669106, 15.9309978, -43.9960480, 16.0249138, -57.0084381, 57.1595917
34: -50.6098480, -4.3519535, -50.7579575, -4.2065744, -42.2331772, 42.2740479
35: -40.8182716, 6.9679499, -41.0102997, 7.0770411, -43.7640457, 43.8656693
36: -44.3910027, 5.2913156, -44.4936447, 5.5051103, -45.5249634, 45.4373016
37: -59.3761292, 2.2931237, -59.4648247, 2.4283562, -55.1317368, 55.1433640
38: -50.8067245, 8.4819918, -50.9487762, 8.6827278, -59.4894524, 59.4307671
39: -52.0627289, 14.8448610, -52.1844101, 14.9468937, -67.0096207, 67.0292740
40: -47.7683372, 8.2927904, -47.8986702, 8.4253368, -53.1812744, 53.1575470
41: -31.7798367, 15.1580048, -31.9125137, 15.3666687, -45.5376892, 45.4534912
42: -27.1065788, 9.9747896, -27.2091503, 10.2502966, -36.8151703, 36.6062660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6884387, upper bound: 14.8130238
time: 35.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6884387, upper bound: 14.7752221
time: 32.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.8448009, 32.9465561, -24.0081558, 32.9007835, -54.3103180, 54.4971619
1: -7.6385298, 32.3773956, -7.7411938, 32.3421783, -36.4696121, 36.5869064
2: -4.6794014, 31.8459244, -4.8735218, 31.8072548, -33.1027527, 33.3405304
3: -8.8700428, 29.0100975, -9.0065479, 28.9244156, -32.3935013, 32.6275558
4: -10.0191402, 35.1108093, -10.2292519, 35.0936279, -43.3408432, 43.5597916
5: -11.0432577, 30.0074749, -11.2195377, 29.9433556, -38.0776520, 38.3372574
6: -38.6985550, 7.4456148, -38.7630997, 7.6009502, -44.5445557, 44.4217606
7: -15.2460032, 30.7052479, -15.3681755, 30.6437511, -41.6713104, 41.8689346
8: -15.3553410, 34.5864143, -15.5648537, 34.5916977, -47.0220947, 47.2329483
9: -10.3158655, 27.0849419, -10.3668442, 27.2356491, -35.8024750, 35.7430534
10: -28.4433041, 23.7239609, -28.5339413, 24.0138168, -50.6327515, 50.4517822
11: -35.7374420, 14.1100473, -35.8387909, 14.3455219, -49.4387131, 49.2999344
12: -49.4948959, 1.7599149, -49.4279556, 2.0895166, -44.2420044, 43.8122101
13: -28.7991905, 21.2953606, -28.8493557, 21.3684311, -49.5815582, 49.5667725
14: -71.0472260, -6.5339088, -71.1331940, -6.3187218, -64.7285004, 64.5992889
15: -17.3247032, 24.6433983, -17.4522133, 24.7016945, -42.0263977, 42.0956116
16: -27.4387398, 23.5336304, -27.5791168, 23.7017136, -48.4303970, 48.4238892
17: -71.2341309, -3.9649734, -71.3407822, -3.7658482, -67.4682846, 67.3758087
18: -34.8268051, 11.6664600, -34.7916565, 11.6926804, -40.6694183, 40.6050682
19: -25.7155037, 5.2232428, -25.7398338, 5.2912602, -29.9502869, 29.9136887
20: -26.4732990, 4.2942748, -26.4865341, 4.3990378, -29.1636314, 29.0548973
21: -31.3745937, 10.0413408, -31.3962822, 10.1789551, -40.4614868, 40.3339233
22: -33.6352654, 6.9391332, -33.6487122, 7.0035205, -38.5023346, 38.4813004
23: -26.9204159, 8.8448753, -26.9388237, 8.9170961, -35.2894135, 35.2531128
24: -23.2304173, 9.8578224, -23.3159676, 9.8816357, -32.7790375, 32.8553772
25: -29.2161217, 6.0586343, -29.2641678, 6.1367998, -34.3957596, 34.3857574
26: -43.1002846, 7.7096410, -42.9769325, 7.8593407, -43.9070053, 43.6534500
27: -26.6267300, 11.4775934, -26.6715012, 11.5451527, -37.8149719, 37.7995758
28: -29.6223068, 7.1594968, -29.6173363, 7.2081265, -36.6627579, 36.6264954
29: -32.6294174, 8.9547062, -32.6316605, 9.0399208, -41.6693382, 41.5863647
30: -37.6100197, 6.9053535, -37.6099358, 7.0039663, -44.6139870, 44.5152893
31: -31.4203033, 7.2620621, -31.4497528, 7.3101988, -37.6264420, 37.6327858
32: -33.6787376, 6.6701479, -33.7355385, 6.9019251, -40.5806618, 40.4056854
33: -43.8438377, 16.0212421, -44.0136986, 16.0647736, -57.1267242, 57.2520218
34: -50.6606903, -4.2402763, -50.7646065, -4.1576748, -42.3470459, 42.3532639
35: -40.8774643, 7.0787134, -41.0198059, 7.1257582, -43.8847885, 43.9455109
36: -44.4465408, 5.4171438, -44.5008926, 5.5604863, -45.6445999, 45.5440826
37: -59.4441872, 2.3598638, -59.4802933, 2.4573307, -55.2396469, 55.1737289
38: -50.8815079, 8.5870695, -50.9598770, 8.7280130, -59.6095200, 59.5469475
39: -52.1407051, 14.8686666, -52.2052460, 14.9565773, -67.0972824, 67.0739136
40: -47.8279457, 8.3259153, -47.9126053, 8.4390574, -53.2527161, 53.1997375
41: -31.8287468, 15.2328167, -31.9230423, 15.3986750, -45.6183319, 45.5309982
42: -27.1428337, 10.0633316, -27.2149715, 10.2881393, -36.9078140, 36.6905899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7332974, upper bound: 14.8130238
time: 30.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.7332974, upper bound: 14.7752221
time: 17.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.6832714, 32.8237000, -23.8903542, 33.0145798, -54.2405548, 54.2526703
1: -7.5349503, 32.2881699, -7.6468420, 32.3715897, -36.4137344, 36.4077644
2: -4.6082377, 31.7449627, -4.7461557, 31.8878593, -33.0778809, 33.1042862
3: -8.8096848, 28.8775425, -8.9426956, 29.0806313, -32.4522095, 32.4084511
4: -9.9323063, 35.0178795, -10.0820522, 35.1476402, -43.2856369, 43.3193893
5: -11.0005951, 29.8810844, -11.1488428, 30.1046734, -38.1544113, 38.1249237
6: -38.6627731, 7.3678131, -38.7160530, 7.4547081, -44.3887329, 44.3746262
7: -15.0863647, 30.5765133, -15.2672386, 30.7065830, -41.5973816, 41.6205521
8: -15.2693176, 34.5076332, -15.4230404, 34.6558685, -46.9696960, 46.9958267
9: -10.2029343, 27.1451054, -10.3383322, 27.2233391, -35.7031174, 35.6705399
10: -28.3520851, 23.8695030, -28.6210365, 23.9857922, -50.5244293, 50.4755020
11: -35.7508888, 14.2376451, -35.9929161, 14.3109188, -49.4057083, 49.5227585
12: -49.3519020, 1.7786703, -49.6777878, 2.0103431, -43.9895401, 44.0660934
13: -28.7698479, 21.2952919, -28.8101654, 21.4120140, -49.6031036, 49.5010147
14: -70.8536301, -6.4705620, -71.1376648, -6.3838921, -64.4697418, 64.6670990
15: -17.3412476, 24.6509838, -17.4125938, 24.7685223, -42.1097717, 42.0635757
16: -27.3600407, 23.6191635, -27.5470257, 23.6769352, -48.3413239, 48.3586807
17: -71.1052322, -3.9124870, -71.2756348, -3.8565063, -67.2487259, 67.3631439
18: -34.7237091, 11.6255989, -34.9027100, 11.6661177, -40.5562515, 40.6883278
19: -25.6690407, 5.2435284, -25.7966862, 5.2655649, -29.8639297, 29.9816170
20: -26.4469795, 4.3407965, -26.5737381, 4.3802323, -29.0635071, 29.1402702
21: -31.3169880, 10.1079197, -31.5187969, 10.1558933, -40.3712463, 40.4410095
22: -33.5647202, 6.9021187, -33.6818123, 6.9765835, -38.3938141, 38.4711380
23: -26.8816051, 8.8507004, -27.0092678, 8.8909073, -35.1886215, 35.2795486
24: -23.2302971, 9.8370571, -23.3070717, 9.8468828, -32.7594223, 32.8538132
25: -29.1709099, 6.0173926, -29.2713242, 6.0821977, -34.2920074, 34.3458481
26: -42.9048920, 7.7195296, -43.2203674, 7.8632226, -43.7182922, 43.8728104
27: -26.6020279, 11.4361534, -26.6853695, 11.4773846, -37.7090378, 37.7611237
28: -29.5621796, 7.0959892, -29.6518211, 7.1580286, -36.5610580, 36.6124649
29: -32.5625076, 8.9324312, -32.6637344, 9.0011635, -41.5636711, 41.5961647
30: -37.5673676, 6.9270239, -37.6992645, 6.9859896, -44.5533562, 44.6262894
31: -31.3622017, 7.2452283, -31.5261497, 7.2623968, -37.4892578, 37.7317390
32: -33.6431885, 6.6632557, -33.7157288, 6.7431860, -40.3863754, 40.3789825
33: -43.8829536, 15.9290857, -43.9940796, 16.0890694, -57.1362457, 57.1349335
34: -50.6621475, -4.3609734, -50.7107201, -4.2182713, -42.2299347, 42.2131500
35: -40.9114838, 6.9653702, -40.9867134, 7.1273093, -43.8499527, 43.8409500
36: -44.3982391, 5.3047290, -44.4218330, 5.4136705, -45.4463806, 45.3817444
37: -59.3787804, 2.3068924, -59.4869156, 2.3709545, -55.1026764, 55.1848831
38: -50.8324890, 8.5125132, -50.9076157, 8.6132641, -59.4457550, 59.4201279
39: -52.0895767, 14.8793678, -52.1842003, 14.9439573, -67.0335312, 67.0635681
40: -47.8012733, 8.2896061, -47.8653908, 8.3463106, -53.1337891, 53.1241074
41: -31.8194580, 15.1884575, -31.8735046, 15.2526960, -45.4514542, 45.4508362
42: -27.1392670, 10.0929079, -27.2275429, 10.1779842, -36.7642593, 36.7993164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6281871, upper bound: 14.8130234
time: 38.20 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6913915, upper bound: 14.7752221
time: 56.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.8972111, 32.8823776, -23.9834328, 33.0178108, -54.4529037, 54.4069977
1: -7.6743460, 32.3303680, -7.7079201, 32.3740921, -36.5458832, 36.5159531
2: -4.7411251, 31.7825947, -4.8046670, 31.8911552, -33.2003250, 33.2053986
3: -8.9097795, 28.9115391, -8.9861326, 29.0850048, -32.5341568, 32.4940567
4: -10.1101875, 35.0665054, -10.1593618, 35.1510162, -43.4538803, 43.4473114
5: -11.1158705, 29.9267635, -11.1991100, 30.1103439, -38.2577972, 38.2292633
6: -38.7067108, 7.4683561, -38.7226982, 7.4976997, -44.4826813, 44.4712372
7: -15.2728996, 30.6248055, -15.3489418, 30.7088966, -41.7532959, 41.7627258
8: -15.4481335, 34.5631256, -15.5003967, 34.6614456, -47.1427917, 47.1363602
9: -10.3429317, 27.1900444, -10.3992395, 27.2273750, -35.8267899, 35.7808342
10: -28.4573860, 23.9261646, -28.6664886, 23.9951401, -50.6305084, 50.5785065
11: -35.7851486, 14.2779970, -36.0066833, 14.3216143, -49.4508820, 49.5819702
12: -49.4042282, 1.9525642, -49.6826019, 2.0841284, -44.1311722, 44.1913376
13: -28.8079758, 21.3283195, -28.8259106, 21.4242477, -49.6537094, 49.5725708
14: -70.9933167, -6.4160938, -71.1994247, -6.3759193, -64.6174011, 64.7833328
15: -17.4158096, 24.6830502, -17.4432411, 24.7736168, -42.1894264, 42.1262894
16: -27.5176601, 23.6700859, -27.6155720, 23.6822319, -48.4864349, 48.4819183
17: -71.1548386, -3.8782883, -71.2962341, -3.8499069, -67.3049316, 67.4179459
18: -34.7488403, 11.6560459, -34.9105988, 11.6782446, -40.5834389, 40.7397194
19: -25.7031803, 5.2575760, -25.8070946, 5.2713766, -29.9141693, 30.0067024
20: -26.4725990, 4.3571301, -26.5805969, 4.3878975, -29.1290855, 29.1571579
21: -31.3614388, 10.1265707, -31.5337811, 10.1632900, -40.4353104, 40.4677811
22: -33.6135597, 6.9618554, -33.6935272, 7.0023942, -38.4705353, 38.5309982
23: -26.9108925, 8.8871193, -27.0170650, 8.9059830, -35.2449646, 35.3265533
24: -23.2554932, 9.8549070, -23.3161621, 9.8560143, -32.7977753, 32.8812866
25: -29.2130928, 6.0950913, -29.2803688, 6.1162553, -34.3704681, 34.4298401
26: -42.9497910, 7.7672834, -43.2297974, 7.8839493, -43.7850647, 43.9159317
27: -26.6332150, 11.4729319, -26.6960049, 11.4883938, -37.7559662, 37.8134766
28: -29.5995121, 7.1646552, -29.6593571, 7.1879025, -36.6302185, 36.6892471
29: -32.6032562, 8.9924936, -32.6743317, 9.0273228, -41.6305771, 41.6668243
30: -37.5902977, 6.9637289, -37.7079926, 6.9996815, -44.5899811, 44.6717224
31: -31.4077873, 7.2758732, -31.5376587, 7.2748690, -37.5775108, 37.7742386
32: -33.6860161, 6.7541246, -33.7229691, 6.7833796, -40.4693947, 40.4770927
33: -43.9598389, 16.0192890, -44.0116806, 16.1288700, -57.2544861, 57.2274246
34: -50.7129669, -4.2492661, -50.7174225, -4.1693888, -42.3438110, 42.2924500
35: -40.9707489, 7.0761113, -40.9962349, 7.1760125, -43.9706726, 43.9207916
36: -44.4538498, 5.4305568, -44.4291725, 5.4690351, -45.5660553, 45.4886398
37: -59.4468765, 2.3735728, -59.5023346, 2.3998494, -55.2104187, 55.2152176
38: -50.9073639, 8.6175661, -50.9187393, 8.6584845, -59.5658493, 59.5363045
39: -52.1676064, 14.9031496, -52.2050133, 14.9536657, -67.1212692, 67.1081619
40: -47.8608627, 8.3226929, -47.8794174, 8.3600330, -53.2053223, 53.1663132
41: -31.8684063, 15.2633705, -31.8840733, 15.2847166, -45.5320740, 45.5283432
42: -27.1755009, 10.1814671, -27.2333622, 10.2157898, -36.8568954, 36.8836517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=224, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6281871, upper bound: 14.8130234
time: 45.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6913915, upper bound: 14.7752221
time: 16.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -23.6854362, 32.8309174, -23.9639893, 33.0445747, -54.2738647, 54.3334045
1: -7.5359058, 32.2962265, -7.6980562, 32.3992233, -36.4386978, 36.4686089
2: -4.6100779, 31.7608261, -4.8387852, 31.9365616, -33.1259689, 33.2127609
3: -8.8115768, 28.8798981, -8.9907370, 29.0969868, -32.4680786, 32.4619370
4: -9.9339342, 35.0332031, -10.1741018, 35.1975708, -43.3361588, 43.4256516
5: -11.0024805, 29.8852882, -11.1950054, 30.1270924, -38.1767197, 38.1843719
6: -38.6903763, 7.3691902, -38.8038788, 7.5887403, -44.5498657, 44.4588394
7: -15.0878849, 30.5876808, -15.3276587, 30.7414474, -41.6327820, 41.6925507
8: -15.2713089, 34.5228653, -15.5050707, 34.7057800, -47.0184479, 47.0934296
9: -10.2003679, 27.1470299, -10.3671398, 27.2573719, -35.7371521, 35.7060280
10: -28.3542480, 23.8809090, -28.7002258, 24.0376015, -50.5780029, 50.6068802
11: -35.7522888, 14.2471380, -36.0405121, 14.3507452, -49.4478912, 49.5788651
12: -49.3618317, 1.7805281, -49.7162018, 2.0542870, -44.0502167, 44.1101074
13: -28.7728157, 21.2957401, -28.8516064, 21.4394798, -49.6306229, 49.5527954
14: -70.8557434, -6.4452114, -71.2789154, -6.3085251, -64.5472183, 64.8337021
15: -17.3426704, 24.6531124, -17.4463081, 24.7851486, -42.1278191, 42.0994186
16: -27.3612347, 23.6222820, -27.6046715, 23.7034512, -48.3699646, 48.4206085
17: -71.1100998, -3.8723755, -71.4664764, -3.7434063, -67.3666916, 67.5941010
18: -34.7253304, 11.6240330, -34.9456100, 11.6898518, -40.5868835, 40.7273216
19: -25.6706066, 5.2466769, -25.8302402, 5.2886806, -29.8872375, 30.0125008
20: -26.4471092, 4.3420181, -26.5842018, 4.4033289, -29.0853386, 29.1553307
21: -31.3178062, 10.1106339, -31.5491676, 10.1793156, -40.3952026, 40.4687958
22: -33.5666656, 6.9049144, -33.7168732, 7.0017133, -38.4196625, 38.5287018
23: -26.8828583, 8.8529930, -27.0358276, 8.9152660, -35.2150497, 35.3075409
24: -23.2314415, 9.8427258, -23.3638687, 9.8831291, -32.7955704, 32.9144363
25: -29.1712685, 6.0221672, -29.3202705, 6.1170535, -34.3289795, 34.4057579
26: -42.9065094, 7.7188473, -43.2460976, 7.8763213, -43.7390137, 43.9039536
27: -26.6107864, 11.4370461, -26.7314606, 11.5413437, -37.8022385, 37.8113899
28: -29.5629215, 7.0981522, -29.6703205, 7.2004580, -36.6095047, 36.6332855
29: -32.5635071, 8.9353275, -32.6949081, 9.0260077, -41.5895157, 41.6302338
30: -37.5672836, 6.9297457, -37.7137527, 7.0146255, -44.5819092, 44.6434975
31: -31.3645668, 7.2551627, -31.5648746, 7.3029175, -37.5260124, 37.7712250
32: -33.6704330, 6.6640720, -33.8059845, 6.8800497, -40.5504837, 40.4700546
33: -43.8855133, 15.9304104, -44.0182190, 16.1327820, -57.1968994, 57.1677551
34: -50.6791000, -4.3601618, -50.7726479, -4.1276045, -42.3461151, 42.2758102
35: -40.9201508, 6.9661336, -41.0241127, 7.1749735, -43.9136124, 43.8769226
36: -44.4259720, 5.3054876, -44.5097160, 5.5365734, -45.5972290, 45.4695282
37: -59.3897362, 2.3085160, -59.5339661, 2.4540396, -55.2072906, 55.2343826
38: -50.8533478, 8.5139084, -50.9887619, 8.7181416, -59.5714874, 59.5026703
39: -52.0972786, 14.8803492, -52.2285576, 14.9921503, -67.0894318, 67.1089096
40: -47.8232689, 8.2909164, -47.9375381, 8.4647427, -53.2742462, 53.1958160
41: -31.8445797, 15.1895561, -31.9596653, 15.3859558, -45.6109619, 45.5360565
42: -27.1568947, 10.0942907, -27.2900105, 10.2782669, -36.8820572, 36.8599625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=177, inp2_unstable=177, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=223, inp2_unstable=224, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6535251, upper bound: 14.8130234
time: 34.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -14.6535251, upper bound: 14.7752221
time: 36.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 73.62 seconds
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6281871, upper bound: 14.8130234
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6281871, upper bound: 14.8130238
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6572734, upper bound: 14.8130234
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6572734, upper bound: 14.7752221
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.7021407, upper bound: 14.7752221
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.7021407, upper bound: 14.7752221
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6593509, upper bound: 14.8130234
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6593509, upper bound: 14.7752221
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.7042232, upper bound: 14.8130234
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.7042232, upper bound: 14.7752221
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6884387, upper bound: 14.8130238
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6884387, upper bound: 14.7752221
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.7332974, upper bound: 14.8130238
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.7332974, upper bound: 14.7752221
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6281871, upper bound: 14.8130234
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6913915, upper bound: 14.7752221
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6281871, upper bound: 14.8130234
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6913915, upper bound: 14.7752221
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6535251, upper bound: 14.8130234
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 73.62
Output dim: 2, lower bound: -14.6535251, upper bound: 14.7752221
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 73.62
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 73.62
Output dim: 2, lower bound: -14.8130228, upper bound: 14.7711210
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 73.62
Output dim: 2, lower bound: -14.8130232, upper bound: 14.7711210

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 49.83 + 3552.18 = 3602.01 seconds
