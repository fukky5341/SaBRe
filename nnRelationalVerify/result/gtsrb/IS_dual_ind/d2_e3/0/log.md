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
execution time: IAR + RelationalAnalysis = 2.33 + 48.61 = 50.94 seconds
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7831672, upper bound: 14.8227297
time: 43.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8227295, upper bound: 14.8227297
time: 41.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 85.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 85.86
Output dim: 2, lower bound: -14.7831672, upper bound: 14.8227297
IS_A2, status: Status.UNKNOWN, split count: 1, time: 85.86
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

Time for backsubstitution: 1.89 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
time: 39.53 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455
time: 19.81 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
time: 20.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455
time: 41.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 63.82 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 63.82
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 63.82
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 63.82
Output dim: 2, lower bound: -14.7812814, upper bound: 14.7829704
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 63.82
Output dim: 2, lower bound: -14.7812814, upper bound: 14.8208455

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -23.7332554, 32.8043785, -23.7412682, 32.8218193, -54.1236954, 54.1081390
1: -7.5913811, 32.3191414, -7.5942326, 32.3065186, -36.3982773, 36.4189110
2: -4.6115685, 31.7183914, -4.6265125, 31.7534904, -32.9970322, 32.9742279
3: -8.8208122, 28.8344116, -8.8503876, 28.8636093, -32.3137436, 32.3234482
4: -9.9515591, 35.0057411, -9.9689713, 35.0254402, -43.2154388, 43.2164307
5: -10.9909039, 29.8198261, -11.0380754, 29.8672543, -37.9757919, 37.9759140
6: -38.6562080, 7.3652134, -38.6805496, 7.3296719, -44.1712646, 44.2827072
7: -15.1730404, 30.6098499, -15.1922121, 30.5931358, -41.5869675, 41.6169662
8: -15.2829733, 34.4699173, -15.2847509, 34.4984283, -46.8677979, 46.8416824
9: -10.2290096, 27.0683746, -10.2665958, 27.1203060, -35.6050491, 35.6022873
10: -28.2017193, 23.7101364, -28.3816109, 23.8432693, -50.1777954, 50.2416916
11: -35.5208893, 14.0722752, -35.7207031, 14.1486654, -49.0108871, 49.1215668
12: -49.2173271, 1.7005596, -49.3602638, 1.8161402, -43.7082138, 43.7411575
13: -28.7572021, 21.2045879, -28.7397118, 21.2555542, -49.4040222, 49.3491974
14: -70.7806549, -6.5390663, -70.8377304, -6.4676495, -64.3130035, 64.2986603
15: -17.2905273, 24.5670662, -17.3187637, 24.6459045, -41.9364319, 41.8858299
16: -27.3242912, 23.5409107, -27.4355488, 23.6034393, -48.2682648, 48.2806931
17: -71.0798187, -3.9750175, -71.1124268, -3.8760891, -67.2037277, 67.1374054
18: -34.6634483, 11.6476955, -34.7050934, 11.6305618, -40.4276314, 40.4824905
19: -25.6118984, 5.2060633, -25.6513844, 5.2123432, -29.7977142, 29.8123055
20: -26.3684464, 4.2592492, -26.4296951, 4.2783332, -28.9316711, 28.9575615
21: -31.2019253, 10.0029488, -31.2998009, 10.0270357, -40.1058655, 40.1906586
22: -33.5581856, 6.8850379, -33.5532761, 6.8803792, -38.3295670, 38.3318787
23: -26.8120804, 8.8126030, -26.8622208, 8.8221245, -35.1224594, 35.1262436
24: -23.1714706, 9.8352566, -23.1924934, 9.8256941, -32.6867294, 32.6912689
25: -29.1510830, 6.0200548, -29.1588764, 6.0228915, -34.2620468, 34.2329865
26: -42.8281860, 7.6498361, -42.8772163, 7.6979380, -43.4871445, 43.4980545
27: -26.5607128, 11.4439278, -26.5807705, 11.4177608, -37.6100464, 37.6509476
28: -29.5614738, 7.1061158, -29.5440292, 7.0901260, -36.5053101, 36.4849014
29: -32.5573425, 8.9021769, -32.5488815, 8.8908825, -41.4482269, 41.4510574
30: -37.5036278, 6.8365421, -37.5256081, 6.8380489, -44.3416748, 44.3621521
31: -31.2905807, 7.2511358, -31.3571835, 7.2551990, -37.5187988, 37.5030136
32: -33.5999794, 6.6252565, -33.6682053, 6.6441889, -40.2441673, 40.2934608
33: -43.8166046, 15.8970661, -43.8643570, 15.9682674, -56.9855957, 56.9705505
34: -50.6480522, -4.3532996, -50.6651840, -4.3294086, -42.1734161, 42.1777267
35: -40.8671951, 6.9572964, -40.9008636, 7.0169969, -43.7602386, 43.7445068
36: -44.4357529, 5.3534050, -44.4292297, 5.3495731, -45.4318161, 45.4324036
37: -59.3718796, 2.3285379, -59.3764343, 2.3370237, -55.0620193, 55.0497894
38: -50.8492393, 8.5426531, -50.8694038, 8.5639210, -59.4131622, 59.4120560
39: -52.0847588, 14.8282928, -52.0914459, 14.8806753, -66.9654312, 66.9197388
40: -47.7853470, 8.2833309, -47.8290024, 8.2904224, -53.0335999, 53.0769882
41: -31.7833595, 15.1797733, -31.8323841, 15.1633453, -45.3237991, 45.3970642
42: -27.0710316, 10.0179863, -27.1534920, 10.0560598, -36.5142059, 36.5919113

Time for backsubstitution: 1.89 seconds

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
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 811
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
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
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
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.7810892
time: 33.06 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7769931, upper bound: 14.7810892
time: 43.38 seconds

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

Time for backsubstitution: 1.91 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
time: 75.96 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
time: 39.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -23.8732204, 32.8977127, -23.7916107, 32.8386078, -54.2497711, 54.2547913
1: -7.6619291, 32.3402443, -7.6195173, 32.3083344, -36.4834595, 36.4542427
2: -4.7378902, 31.7998352, -4.6817389, 31.7558212, -33.0680389, 33.1169968
3: -8.9360828, 28.9148445, -8.9001846, 28.8738937, -32.3749008, 32.4291916
4: -10.0912485, 35.0844383, -10.0303688, 35.0317116, -43.3225479, 43.3385849
5: -11.1479568, 29.9328175, -11.1071939, 29.8825874, -38.0804825, 38.1586838
6: -38.7455750, 7.4078732, -38.7076378, 7.3450909, -44.3742371, 44.3794556
7: -15.2751637, 30.6400242, -15.2273312, 30.6009216, -41.6763306, 41.6836929
8: -15.4269447, 34.5783234, -15.3499126, 34.5087280, -46.9665985, 47.0157242
9: -10.3161163, 27.2131367, -10.2707977, 27.1848278, -35.7613449, 35.6643562
10: -28.4319344, 24.0093460, -28.3912678, 23.9853935, -50.5524368, 50.3811874
11: -35.7928009, 14.3025360, -35.7474594, 14.2611589, -49.3898468, 49.3103027
12: -49.4164352, 2.0172338, -49.3710518, 1.9629469, -44.0586777, 44.0013275
13: -28.7912865, 21.3417130, -28.7469139, 21.3048096, -49.4975739, 49.4961777
14: -70.9319611, -6.3240738, -70.8480072, -6.3690300, -64.5629272, 64.5239334
15: -17.4022427, 24.6919136, -17.3657322, 24.6672230, -42.0694656, 42.0576477
16: -27.4994354, 23.6915627, -27.4548454, 23.6768436, -48.4864502, 48.3677292
17: -71.1527023, -3.7657852, -71.1246796, -3.7798309, -67.3728714, 67.3588943
18: -34.7499542, 11.6669884, -34.7226524, 11.6368904, -40.5347366, 40.5424919
19: -25.7050629, 5.2629476, -25.6695576, 5.2404370, -29.8763123, 29.8800583
20: -26.4776382, 4.3543711, -26.4457874, 4.3220463, -29.0560760, 29.0070152
21: -31.3637352, 10.1337490, -31.3180580, 10.0905075, -40.3499146, 40.2392197
22: -33.6151886, 6.9540567, -33.5721092, 6.9064221, -38.4121323, 38.4153671
23: -26.9111500, 8.8823624, -26.8735580, 8.8525906, -35.1868210, 35.1801453
24: -23.2627125, 9.8534603, -23.2252464, 9.8290386, -32.7810364, 32.7843704
25: -29.2154503, 6.0941563, -29.1744118, 6.0537467, -34.3122787, 34.3364639
26: -42.9554367, 7.8141112, -42.8989143, 7.7696815, -43.7054138, 43.6399612
27: -26.6536102, 11.4573975, -26.6148167, 11.4213753, -37.7179718, 37.7087936
28: -29.6049652, 7.1389256, -29.5578957, 7.0987010, -36.5499268, 36.5699539
29: -32.6070099, 8.9796715, -32.5586586, 8.9248924, -41.5319023, 41.5383301
30: -37.5948486, 6.9379950, -37.5430717, 6.8800058, -44.4748535, 44.4810677
31: -31.4130478, 7.2905550, -31.3784142, 7.2735834, -37.5727959, 37.6434250
32: -33.7141342, 6.7389078, -33.6812286, 6.6946917, -40.4088249, 40.4201355
33: -43.9889755, 16.0096016, -43.9419823, 15.9833641, -57.1206512, 57.1600037
34: -50.7490540, -4.2797999, -50.7102165, -4.3229027, -42.2452011, 42.2897949
35: -41.0058289, 7.0560403, -40.9649124, 7.0261049, -43.8605881, 43.9051514
36: -44.4875259, 5.4056673, -44.4496155, 5.3638301, -45.5096893, 45.5105286
37: -59.4601059, 2.3753119, -59.4070587, 2.3513141, -55.1807709, 55.1646271
38: -50.9411316, 8.6145859, -50.9044685, 8.5885544, -59.5296860, 59.5190544
39: -52.1699600, 14.9148359, -52.1212921, 14.9091911, -67.0791473, 67.0361252
40: -47.8925934, 8.3204975, -47.8672333, 8.2984991, -53.1475677, 53.1512680
41: -31.9004517, 15.2329741, -31.8608971, 15.1856747, -45.4720230, 45.4762878
42: -27.1971684, 10.1639338, -27.1693192, 10.1237459, -36.7766342, 36.7816887

Time for backsubstitution: 1.89 seconds

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
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 546
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
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.7810892
time: 40.15 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.7810892
time: 45.78 seconds

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

Time for backsubstitution: 1.91 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
time: 42.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8189531, upper bound: 14.8189535
time: 81.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 126.45 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 126.45
Output dim: 2, lower bound: -14.7458382, upper bound: 14.7810892
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 126.45
Output dim: 2, lower bound: -14.7769931, upper bound: 14.7810892
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 126.45
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 126.45
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 126.45
Output dim: 2, lower bound: -14.7458382, upper bound: 14.7810892
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 126.45
Output dim: 2, lower bound: -14.7458382, upper bound: 14.7810892
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 126.45
Output dim: 2, lower bound: -14.7458382, upper bound: 14.8189535
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 126.45
Output dim: 2, lower bound: -14.8189531, upper bound: 14.8189535

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -23.6825638, 32.7970695, -23.7295895, 32.8201942, -54.0707397, 54.8483429
1: -7.5673695, 32.3171997, -7.5887756, 32.3060608, -36.1502380, 36.4104805
2: -4.5654535, 31.7162933, -4.6159477, 31.7530193, -32.8558578, 32.9610519
3: -8.7640524, 28.8297882, -8.8368607, 28.8625526, -31.8773270, 32.3060150
4: -9.9161234, 35.0018921, -9.9609060, 35.0245476, -43.1809540, 43.3429260
5: -10.9262114, 29.8129730, -11.0228863, 29.8656712, -37.9101562, 38.1155701
6: -38.6421623, 7.3576717, -38.6773911, 7.3279514, -43.4320145, 44.2659531
7: -15.1287603, 30.6053104, -15.1820164, 30.5920963, -40.3487244, 41.6019897
8: -15.2376614, 34.4648857, -15.2744217, 34.4972954, -47.1166534, 46.8262863
9: -10.2238426, 27.0450974, -10.2654305, 27.1149940, -35.5947723, 35.9615936
10: -28.1958027, 23.6367950, -28.3802795, 23.8266602, -50.1139069, 50.1672211
11: -35.5061378, 14.0189199, -35.7173233, 14.1365738, -49.5041656, 49.0639114
12: -49.2115669, 1.5898085, -49.3589439, 1.7908559, -43.5613174, 43.6287384
13: -28.7420444, 21.1829510, -28.7362690, 21.2505569, -49.3839340, 49.9192200
14: -70.7711258, -6.6081409, -70.8356018, -6.4832497, -64.2878723, 64.2274628
15: -17.2725811, 24.5548725, -17.3146648, 24.6432114, -41.9157944, 41.8695374
16: -27.3106976, 23.5198803, -27.4324989, 23.5986881, -47.6970749, 48.2628632
17: -71.0701904, -4.0435448, -71.1102142, -3.8917885, -67.1784058, 67.0666656
18: -34.6555748, 11.6175556, -34.7033081, 11.6237373, -40.4116516, 40.2834816
19: -25.6016617, 5.1831970, -25.6490784, 5.2070503, -29.7823524, 29.5777092
20: -26.3592072, 4.2318263, -26.4276104, 4.2721119, -29.3010979, 28.9313622
21: -31.1895676, 9.9572525, -31.2969971, 10.0166969, -40.2163849, 40.1432953
22: -33.5491486, 6.8501139, -33.5511932, 6.8724928, -38.3124619, 36.6681137
23: -26.8044777, 8.7956076, -26.8604832, 8.8182793, -35.1099014, 34.7938309
24: -23.1596756, 9.8248291, -23.1897945, 9.8233442, -32.6722603, 32.4159241
25: -29.1424675, 5.9952579, -29.1569138, 6.0171146, -34.2482834, 32.8055267
26: -42.8166771, 7.5588179, -42.8746338, 7.6773491, -43.6809464, 43.4049683
27: -26.5455971, 11.4278946, -26.5773258, 11.4141550, -37.9597511, 37.6322632
28: -29.5524845, 7.0968475, -29.5419865, 7.0880232, -36.4941177, 36.3444748
29: -32.5497169, 8.8673992, -32.5471420, 8.8829718, -41.4326897, 41.4145432
30: -37.4930954, 6.8120975, -37.5231743, 6.8322897, -44.3253860, 44.3352737
31: -31.2782593, 7.2359252, -31.3543720, 7.2517409, -37.5022888, 37.0499420
32: -33.5937805, 6.6038561, -33.6668053, 6.6392984, -39.9810257, 40.2706604
33: -43.7709846, 15.8875580, -43.8540459, 15.9660587, -56.9379272, 56.9875488
34: -50.6224136, -4.3592958, -50.6594086, -4.3307867, -42.1462173, 42.1338348
35: -40.8325119, 6.9523740, -40.8930283, 7.0158691, -43.7229996, 43.8307419
36: -44.4283180, 5.3391519, -44.4275169, 5.3463349, -45.4189606, 45.4991150
37: -59.3579407, 2.3132715, -59.3733215, 2.3335767, -55.0399628, 55.4144287
38: -50.8344307, 8.5259762, -50.8660469, 8.5601606, -59.3945923, 59.3920212
39: -52.0706367, 14.8129377, -52.0881996, 14.8770733, -66.9477081, 66.9011383
40: -47.7635460, 8.2780037, -47.8240051, 8.2892532, -52.9353333, 53.0671387
41: -31.7695427, 15.1693916, -31.8292923, 15.1609669, -44.9566650, 45.3800583
42: -27.0617180, 10.0003834, -27.1513863, 10.0515137, -36.4955673, 36.5570984

Time for backsubstitution: 1.91 seconds

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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1431
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

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.7768124
time: 18.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.7768124
time: 39.50 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -23.7677555, 32.9473724, -23.7327900, 32.8206139, -54.1582947, 54.2502899
1: -7.6008816, 32.3772659, -7.5883255, 32.3058205, -36.4137115, 36.4658775
2: -4.6263514, 31.8451729, -4.6218252, 31.7525063, -33.0109406, 33.1048851
3: -8.8353157, 29.0062904, -8.8444290, 28.8619003, -32.3209229, 32.4831734
4: -9.9627419, 35.1094551, -9.9640732, 35.0239983, -43.2281036, 43.3181992
5: -11.0064144, 30.0024147, -11.0319309, 29.8654633, -37.9856262, 38.1540833
6: -38.6958656, 7.3761387, -38.6771660, 7.3246145, -44.2376556, 44.2836990
7: -15.2016144, 30.7047195, -15.1841478, 30.5907898, -41.6096497, 41.7103653
8: -15.2862663, 34.5813141, -15.2766037, 34.4970055, -46.8743973, 46.9602280
9: -10.2863998, 27.0825729, -10.2644653, 27.1135406, -35.6497345, 35.6228600
10: -28.4098320, 23.7220516, -28.3802986, 23.8336487, -50.3763885, 50.2505798
11: -35.7289200, 14.0717020, -35.7170525, 14.1415014, -49.2150116, 49.1143494
12: -49.4914131, 1.7115173, -49.3575249, 1.8025002, -43.9756165, 43.7446365
13: -28.7647190, 21.2856922, -28.7323399, 21.2519035, -49.4073639, 49.4173126
14: -70.9741364, -6.5381432, -70.8343811, -6.4761505, -64.4979858, 64.2962341
15: -17.2900352, 24.6382751, -17.3049812, 24.6433601, -41.9333954, 41.9432564
16: -27.4059296, 23.5304756, -27.4317169, 23.5949898, -48.2999115, 48.2726288
17: -71.2174072, -3.9637833, -71.1091461, -3.8826256, -67.3347778, 67.1453629
18: -34.8184509, 11.6487303, -34.7030258, 11.6237431, -40.5844612, 40.4937859
19: -25.7055550, 5.2025032, -25.6493797, 5.2091455, -29.8692932, 29.8058052
20: -26.4689770, 4.2627416, -26.4281197, 4.2743077, -29.0323524, 28.9639626
21: -31.3636703, 10.0000534, -31.2969189, 10.0214138, -40.2721634, 40.1860123
22: -33.6259193, 6.8938260, -33.5498657, 6.8698950, -38.3954010, 38.3379822
23: -26.9117813, 8.8207397, -26.8604927, 8.8196754, -35.2005386, 35.1399460
24: -23.2240219, 9.8402157, -23.1896267, 9.8209915, -32.7229309, 32.6962814
25: -29.2093716, 6.0273910, -29.1562939, 6.0190187, -34.2981873, 34.2443352
26: -43.0926971, 7.6658449, -42.8734016, 7.6864920, -43.7432251, 43.5086670
27: -26.6229935, 11.4450331, -26.5779629, 11.4131298, -37.6640854, 37.6505852
28: -29.6179104, 7.1223869, -29.5422668, 7.0871525, -36.5473328, 36.5153275
29: -32.6244469, 8.9039726, -32.5458946, 8.8847475, -41.5091934, 41.4498672
30: -37.6041412, 6.8512125, -37.5229950, 6.8338041, -44.4379463, 44.3742065
31: -31.4109573, 7.2515306, -31.3549862, 7.2527790, -37.5882339, 37.5172424
32: -33.6734314, 6.6325073, -33.6664314, 6.6382475, -40.3116798, 40.2989388
33: -43.8245468, 16.0006256, -43.8580132, 15.9658222, -56.9911804, 57.0812988
34: -50.6540718, -4.2776203, -50.6614532, -4.3309536, -42.1747437, 42.2463608
35: -40.8691635, 7.0530186, -40.8957367, 7.0160704, -43.7589874, 43.8305740
36: -44.4449768, 5.3768239, -44.4268379, 5.3443928, -45.4376526, 45.4571457
37: -59.4322510, 2.3446360, -59.3728638, 2.3294845, -55.1084518, 55.0786819
38: -50.8795700, 8.5648623, -50.8654022, 8.5567636, -59.4363327, 59.4302635
39: -52.1210938, 14.8633232, -52.0884781, 14.8732204, -66.9943161, 66.9517975
40: -47.8156738, 8.3169804, -47.8245621, 8.2874899, -53.0650482, 53.1092834
41: -31.8220444, 15.1906319, -31.8293362, 15.1577568, -45.3727264, 45.3987427
42: -27.1375942, 10.0242653, -27.1511612, 10.0465879, -36.6360626, 36.5946999

Time for backsubstitution: 1.93 seconds

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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 546
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7436501, upper bound: 14.7768124
time: 52.41 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.7768124
time: 37.65 seconds

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

Time for backsubstitution: 1.82 seconds

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
time: 35.05 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
time: 54.91 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7436501, upper bound: 14.8146118
time: 34.20 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7727093, upper bound: 14.8146118
time: 43.14 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.8223381, 32.8904533, -23.7798920, 32.8368912, -54.1964111, 54.2353439
1: -7.6376753, 32.3382301, -7.6140423, 32.3078575, -36.4604492, 36.4454346
2: -4.6899042, 31.7976799, -4.6709108, 31.7553406, -33.0201874, 33.1037598
3: -8.8769789, 28.9100647, -8.8866177, 28.8728256, -32.3183212, 32.4112358
4: -10.0554647, 35.0805206, -10.0222626, 35.0308456, -43.2872314, 43.3266525
5: -11.0808668, 29.9259453, -11.0918722, 29.8810272, -38.0125504, 38.1365128
6: -38.7315979, 7.4002132, -38.7044983, 7.3433189, -44.3544846, 44.3610458
7: -15.2301025, 30.6354675, -15.2169380, 30.5998592, -41.6299896, 41.6686020
8: -15.3810272, 34.5733032, -15.3395472, 34.5076447, -46.9217911, 47.0001297
9: -10.3108997, 27.1896210, -10.2696018, 27.1795063, -35.7508926, 35.6416435
10: -28.4260235, 23.9357147, -28.3899212, 23.9687576, -50.5293579, 50.3065643
11: -35.7780075, 14.2491655, -35.7440720, 14.2491217, -49.3628387, 49.2525711
12: -49.4107018, 1.9060640, -49.3697128, 1.9376431, -44.0273743, 43.8886566
13: -28.7764626, 21.3190346, -28.7434521, 21.2997360, -49.4780884, 49.4705505
14: -70.9223175, -6.3949852, -70.8458176, -6.3847179, -64.5375977, 64.4508362
15: -17.3825779, 24.6800880, -17.3613033, 24.6645279, -42.0471039, 42.0413895
16: -27.4860229, 23.6700745, -27.4518032, 23.6720772, -48.4680176, 48.3488693
17: -71.1430283, -3.8370152, -71.1224670, -3.7959137, -67.3471146, 67.2854538
18: -34.7420387, 11.6367750, -34.7208595, 11.6300459, -40.5186310, 40.5118408
19: -25.6947403, 5.2399139, -25.6672363, 5.2351084, -29.8605728, 29.8581924
20: -26.4684391, 4.3268528, -26.4437027, 4.3158388, -29.0397987, 28.9785538
21: -31.3513565, 10.0880442, -31.3152695, 10.0801477, -40.3272247, 40.1893921
22: -33.6061897, 6.9193549, -33.5700378, 6.8985972, -38.3950500, 38.3772659
23: -26.9035225, 8.8653049, -26.8718414, 8.8487463, -35.1741257, 35.1640472
24: -23.2502670, 9.8430099, -23.2224808, 9.8266678, -32.7661667, 32.7727890
25: -29.2066841, 6.0686769, -29.1724625, 6.0479937, -34.2981339, 34.3153839
26: -42.9439278, 7.7227936, -42.8963013, 7.7490854, -43.6728821, 43.5464630
27: -26.6382389, 11.4412813, -26.6113281, 11.4177551, -37.6980591, 37.6900177
28: -29.5958824, 7.1296782, -29.5558376, 7.0965543, -36.5385818, 36.5603256
29: -32.5993881, 8.9446650, -32.5569229, 8.9169397, -41.5163269, 41.5015869
30: -37.5842819, 6.9123516, -37.5406914, 6.8741951, -44.4584770, 44.4530411
31: -31.4008007, 7.2752457, -31.3756008, 7.2700949, -37.5559349, 37.6296005
32: -33.7079391, 6.7173100, -33.6798439, 6.6897678, -40.3977051, 40.3971558
33: -43.9431992, 16.0000591, -43.9316826, 15.9812155, -57.0728302, 57.1397934
34: -50.7233276, -4.2858868, -50.7044106, -4.3242640, -42.2175446, 42.2778549
35: -40.9710808, 7.0511727, -40.9570923, 7.0250120, -43.8232498, 43.8918610
36: -44.4800034, 5.3910298, -44.4478569, 5.3604827, -45.4966736, 45.4924698
37: -59.4459190, 2.3600159, -59.4038544, 2.3478737, -55.1584473, 55.1431580
38: -50.9262810, 8.5967770, -50.9011421, 8.5845909, -59.5108719, 59.4979172
39: -52.1556854, 14.8987455, -52.1181068, 14.9055538, -67.0612411, 67.0168533
40: -47.8706284, 8.3150959, -47.8622437, 8.2972679, -53.1253662, 53.1414642
41: -31.8867760, 15.2222338, -31.8578072, 15.1832123, -45.4541397, 45.4587479
42: -27.1879330, 10.1438198, -27.1672020, 10.1192045, -36.7592468, 36.7468910

Time for backsubstitution: 1.84 seconds

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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
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
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
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
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1659
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.7768124
time: 45.24 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7378281, upper bound: 14.7768124
time: 32.95 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.9101868, 33.0420685, -23.7833157, 32.8373718, -54.2861862, 54.3993683
1: -7.6715488, 32.3985977, -7.6136932, 32.3076172, -36.5020828, 36.5055313
2: -4.7542620, 31.9306660, -4.6773605, 31.7548714, -33.0826416, 33.2502060
3: -8.9510746, 29.0887928, -8.8943377, 28.8723030, -32.3841019, 32.5968552
4: -10.1012926, 35.1888275, -10.0254478, 35.0303726, -43.3352966, 43.4428406
5: -11.1640558, 30.1185875, -11.1011381, 29.8808384, -38.0923233, 38.3404617
6: -38.7865906, 7.4190683, -38.7044487, 7.3400011, -44.4536591, 44.3793259
7: -15.3045864, 30.7361488, -15.2193661, 30.5986004, -41.7008743, 41.7790070
8: -15.4334402, 34.6950455, -15.3421698, 34.5074158, -46.9749451, 47.1402893
9: -10.3739634, 27.2269173, -10.2686911, 27.1780930, -35.8096313, 35.6840973
10: -28.6410160, 24.0213108, -28.3899879, 23.9757004, -50.7571411, 50.3902283
11: -36.0018616, 14.3018875, -35.7438927, 14.2540588, -49.6045990, 49.3073349
12: -49.7044601, 2.0347567, -49.3683548, 1.9493008, -44.3341827, 44.0064774
13: -28.7999706, 21.4152641, -28.7396355, 21.3011475, -49.5016403, 49.5583191
14: -71.1337051, -6.3180084, -70.8446960, -6.3753166, -64.7583923, 64.5266876
15: -17.4063187, 24.7758198, -17.3540535, 24.6647053, -42.0710220, 42.1298752
16: -27.5867310, 23.6825638, -27.4512005, 23.6684666, -48.5387497, 48.3597794
17: -71.2934799, -3.7517986, -71.1214752, -3.7860813, -67.5074005, 67.3696747
18: -34.9077301, 11.6683083, -34.7206459, 11.6300697, -40.6978416, 40.5574913
19: -25.8020592, 5.2608414, -25.6676006, 5.2372446, -29.9527588, 29.8742104
20: -26.5790710, 4.3585353, -26.4442215, 4.3181171, -29.1675682, 29.0129128
21: -31.5265083, 10.1308479, -31.3152905, 10.0849018, -40.5186615, 40.2350998
22: -33.6818199, 6.9647961, -33.5687103, 6.8960953, -38.4762344, 38.4215240
23: -27.0127163, 8.8903809, -26.8719234, 8.8501530, -35.2739487, 35.1939926
24: -23.3106995, 9.8582878, -23.2225609, 9.8243999, -32.8186340, 32.7903290
25: -29.2756062, 6.1023464, -29.1720219, 6.0498729, -34.3581772, 34.3495560
26: -43.2271843, 7.8382339, -42.8952026, 7.7594566, -43.9692383, 43.6516800
27: -26.7138672, 11.4589500, -26.6120033, 11.4167690, -37.7677841, 37.7089462
28: -29.6622887, 7.1559243, -29.5561752, 7.0958014, -36.5933990, 36.6016312
29: -32.6763306, 8.9825525, -32.5557098, 8.9187469, -41.5950775, 41.5382614
30: -37.7026939, 6.9532804, -37.5405731, 6.8757849, -44.5784798, 44.4938545
31: -31.5355740, 7.2911582, -31.3763008, 7.2711611, -37.6580200, 37.6580124
32: -33.7883301, 6.7462854, -33.6794853, 6.6889868, -40.4773178, 40.4257698
33: -43.9973526, 16.1135445, -43.9356842, 15.9809265, -57.1266785, 57.2713318
34: -50.7556267, -4.2032838, -50.7064934, -4.3243895, -42.2471085, 42.3634033
35: -41.0084534, 7.1522245, -40.9598694, 7.0252190, -43.8599777, 43.9945068
36: -44.4969177, 5.4295917, -44.4472733, 5.3587756, -45.5158081, 45.5353088
37: -59.5217056, 2.3919444, -59.4036636, 2.3437033, -55.2279510, 55.1945190
38: -50.9722481, 8.6389122, -50.9006233, 8.5823231, -59.5545731, 59.5395355
39: -52.2070312, 14.9497147, -52.1184692, 14.9016867, -67.1087189, 67.0681839
40: -47.9231873, 8.3544922, -47.8627472, 8.2955799, -53.1792679, 53.1840210
41: -31.9418907, 15.2449999, -31.8579025, 15.1802626, -45.5239258, 45.4788437
42: -27.2734699, 10.1746950, -27.1670723, 10.1143284, -36.9002228, 36.7845688

Time for backsubstitution: 1.88 seconds

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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
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
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1543
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
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1659
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1776

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7855901, upper bound: 14.7768124
time: 35.31 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8146113, upper bound: 14.7768124
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

Time for backsubstitution: 1.90 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
time: 36.69 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
time: 48.13 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7855901, upper bound: 14.8146118
time: 41.55 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8146113, upper bound: 14.8146118
time: 37.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 81.41 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7125045, upper bound: 14.7768124
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7125045, upper bound: 14.7768124
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7436501, upper bound: 14.7768124
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7125045, upper bound: 14.7768124
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7436501, upper bound: 14.8146118
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7727093, upper bound: 14.8146118
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7125045, upper bound: 14.7768124
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7378281, upper bound: 14.7768124
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7855901, upper bound: 14.7768124
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.8146113, upper bound: 14.7768124
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7125045, upper bound: 14.8146118
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.7855901, upper bound: 14.8146118
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 81.41
Output dim: 2, lower bound: -14.8146113, upper bound: 14.8146118

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -23.6778889, 32.7837257, -23.7187939, 32.7891922, -54.0350494, 54.8232574
1: -7.5650806, 32.3041191, -7.5834694, 32.2759819, -36.1184006, 36.3924408
2: -4.5622702, 31.6964626, -4.6085663, 31.7080154, -32.8072739, 32.9341736
3: -8.7607555, 28.8244209, -8.8292589, 28.8503380, -31.8577423, 32.2913551
4: -9.9131899, 34.9824715, -9.9542341, 34.9796257, -43.1326828, 43.3166428
5: -10.9231377, 29.8046112, -11.0158653, 29.8463287, -37.8838654, 38.0983429
6: -38.6095734, 7.3548403, -38.6029510, 7.3215418, -43.3925858, 44.1880646
7: -15.1260996, 30.5910416, -15.1758366, 30.5595512, -40.3104858, 41.5815277
8: -15.2343102, 34.4462700, -15.2666197, 34.4541817, -47.0693741, 46.7991867
9: -10.2181606, 27.0410843, -10.2523890, 27.1057072, -35.5781631, 35.9440689
10: -28.1927719, 23.6156006, -28.3732681, 23.7777214, -50.0485382, 50.1322784
11: -35.5023193, 14.0049973, -35.7083740, 14.1049843, -49.4599991, 49.0402832
12: -49.1983490, 1.5854516, -49.3286285, 1.7807899, -43.5346832, 43.5871201
13: -28.7367821, 21.1784363, -28.7242012, 21.2400608, -49.3639526, 49.9026375
14: -70.7659912, -6.6402760, -70.8238373, -6.5577469, -64.2082443, 64.1835632
15: -17.2698936, 24.5493488, -17.3085060, 24.6304245, -41.9003181, 41.8578568
16: -27.3073578, 23.5133247, -27.4247513, 23.5838032, -47.6744843, 48.2459869
17: -71.0623474, -4.0933304, -71.0920410, -4.0073471, -67.0550003, 66.9987106
18: -34.6515274, 11.6096725, -34.6939316, 11.6063452, -40.3911362, 40.2665176
19: -25.5978355, 5.1766615, -25.6402912, 5.1920733, -29.7647705, 29.5624352
20: -26.3561516, 4.2292728, -26.4205379, 4.2662215, -29.2892990, 28.9215546
21: -31.1859131, 9.9503698, -31.2886829, 10.0009680, -40.1990280, 40.1284256
22: -33.5452919, 6.8447628, -33.5423355, 6.8600974, -38.2870636, 36.6536255
23: -26.8010960, 8.7892513, -26.8527050, 8.8036737, -35.0929947, 34.7794037
24: -23.1548634, 9.8136435, -23.1787033, 9.7975864, -32.6417999, 32.3937454
25: -29.1392784, 5.9839926, -29.1495552, 5.9914274, -34.2184448, 32.7873039
26: -42.8118629, 7.5561628, -42.8636017, 7.6712627, -43.6671295, 43.3888702
27: -26.5300636, 11.4258213, -26.5413532, 11.4094563, -37.9395218, 37.5905533
28: -29.5462437, 7.0932875, -29.5277901, 7.0798359, -36.4770813, 36.3231735
29: -32.5441933, 8.8628044, -32.5345154, 8.8722706, -41.4164658, 41.3973198
30: -37.4909248, 6.8057461, -37.5181580, 6.8177443, -44.3086700, 44.3239059
31: -31.2731438, 7.2218342, -31.3425922, 7.2193284, -37.4674149, 37.0252991
32: -33.5598183, 6.6007395, -33.5881653, 6.6321554, -39.9401703, 40.1889038
33: -43.7638893, 15.8853092, -43.8376808, 15.9608059, -56.9203033, 56.9666138
34: -50.5982513, -4.3616838, -50.6034355, -4.3362308, -42.1166534, 42.0760651
35: -40.8185921, 6.9513087, -40.8613129, 7.0133209, -43.7062149, 43.7958527
36: -44.3933144, 5.3371906, -44.3465538, 5.3418398, -45.3792496, 45.4158401
37: -59.3384018, 2.3088684, -59.3296318, 2.3234119, -55.0078888, 55.3566666
38: -50.8042068, 8.5232553, -50.7965164, 8.5538549, -59.3580627, 59.3197708
39: -52.0565338, 14.8102913, -52.0560455, 14.8709068, -66.9274445, 66.8663330
40: -47.7355270, 8.2747746, -47.7593346, 8.2818918, -52.8987122, 52.9971771
41: -31.7376728, 15.1664095, -31.7554302, 15.1540728, -44.9177704, 45.3029709
42: -27.0393562, 9.9972353, -27.0996780, 10.0442352, -36.4660645, 36.5019760

Time for backsubstitution: 1.91 seconds

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
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1712
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
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 47.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 42.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.6800594, 32.7909622, -23.7924175, 32.8192406, -54.0685120, 54.9025803
1: -7.5660210, 32.3121796, -7.6346855, 32.3036118, -36.1433792, 36.4532890
2: -4.5641136, 31.7123680, -4.7012267, 31.7567768, -32.8556671, 33.0426369
3: -8.7626553, 28.8268070, -8.8772850, 28.8666763, -31.8729553, 32.3447342
4: -9.9148293, 34.9978371, -10.0462799, 35.0296173, -43.1831970, 43.4227905
5: -10.9250498, 29.8088112, -11.0619650, 29.8687878, -37.9062119, 38.1558228
6: -38.6371689, 7.3561993, -38.6908264, 7.4556274, -43.5535583, 44.2723694
7: -15.1276903, 30.6022224, -15.2361383, 30.5944061, -40.3453140, 41.6534729
8: -15.2362671, 34.4615402, -15.3486958, 34.5041504, -47.1200714, 46.8967438
9: -10.2155867, 27.0429688, -10.2812614, 27.1397438, -35.6121750, 35.9780922
10: -28.1948795, 23.6269951, -28.4524803, 23.8295422, -50.1018448, 50.2636337
11: -35.5037003, 14.0145483, -35.7559776, 14.1447239, -49.4985809, 49.0963669
12: -49.2082748, 1.5872827, -49.3670464, 1.8247333, -43.5916748, 43.6310959
13: -28.7397652, 21.1788483, -28.7656403, 21.2675800, -49.3915558, 49.9444885
14: -70.7680435, -6.6149464, -70.9649963, -6.4823036, -64.2857361, 64.3500519
15: -17.2712955, 24.5514736, -17.3422432, 24.6470985, -41.9183960, 41.8937149
16: -27.3084831, 23.5164394, -27.4823914, 23.6104336, -47.7030029, 48.3078766
17: -71.0671768, -4.0531960, -71.2829742, -3.8942375, -67.1729431, 67.2297821
18: -34.6531525, 11.6081753, -34.7368431, 11.6300449, -40.4216461, 40.3039627
19: -25.5993881, 5.1798139, -25.6737785, 5.2151713, -29.7879410, 29.5921478
20: -26.3562775, 4.2305055, -26.4309826, 4.2893248, -29.3088989, 28.9366150
21: -31.1867485, 9.9531116, -31.3190212, 10.0243855, -40.2224197, 40.1562195
22: -33.5472527, 6.8475542, -33.5773811, 6.8852348, -38.3129807, 36.6934586
23: -26.8023300, 8.7915363, -26.8792553, 8.8279333, -35.1194153, 34.8075714
24: -23.1560059, 9.8193531, -23.2354984, 9.8337994, -32.6779099, 32.4528160
25: -29.1396255, 5.9887662, -29.1984634, 6.0262413, -34.2553635, 32.8412933
26: -42.8135529, 7.5554409, -42.8893738, 7.6843405, -43.6871567, 43.4200211
27: -26.5388565, 11.4267349, -26.5874844, 11.4734192, -38.0122757, 37.6409454
28: -29.5469952, 7.0954165, -29.5463142, 7.1223111, -36.5255814, 36.3443832
29: -32.5452347, 8.8657198, -32.5656662, 8.8970222, -41.4422569, 41.4313850
30: -37.4908257, 6.8085070, -37.5327072, 6.8463078, -44.3371353, 44.3412132
31: -31.2755146, 7.2317605, -31.3812542, 7.2597489, -37.5041809, 37.0634003
32: -33.5870781, 6.6015186, -33.6784782, 6.7689438, -40.1054993, 40.2799988
33: -43.7664986, 15.8866310, -43.8618355, 16.0045300, -56.9806824, 56.9957581
34: -50.6151924, -4.3608699, -50.6652985, -4.2455654, -42.2327728, 42.1389389
35: -40.8272247, 6.9520226, -40.8986702, 7.0609913, -43.7698898, 43.8302917
36: -44.4210510, 5.3379602, -44.4344330, 5.4647651, -45.5301361, 45.5036545
37: -59.3493805, 2.3104854, -59.3765182, 2.4064884, -55.1125488, 55.4044724
38: -50.8250084, 8.5246162, -50.8775253, 8.6587982, -59.4838066, 59.4021416
39: -52.0642853, 14.8111610, -52.1003723, 14.9190111, -66.9832993, 66.9115295
40: -47.7574844, 8.2761183, -47.8314819, 8.4002552, -53.0403442, 53.0687943
41: -31.7627449, 15.1674948, -31.8416481, 15.2873268, -45.0766220, 45.3882904
42: -27.0570297, 9.9986506, -27.1622257, 10.1445894, -36.5836105, 36.5626755

Time for backsubstitution: 1.94 seconds

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
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1712
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
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 42.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7399635, upper bound: 14.7752221
time: 32.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -23.7630844, 32.9339600, -23.7220592, 32.7896461, -54.1226196, 54.2257767
1: -7.5985622, 32.3642082, -7.5830021, 32.2757187, -36.3817902, 36.4478722
2: -4.6231527, 31.8253746, -4.6144676, 31.7075233, -32.9624252, 33.0780029
3: -8.8320179, 29.0010242, -8.8368244, 28.8497028, -32.3027954, 32.4685440
4: -9.9598484, 35.0899887, -9.9573650, 34.9790726, -43.1797867, 43.2917633
5: -11.0033550, 29.9941063, -11.0249052, 29.8460884, -37.9593811, 38.1363907
6: -38.6632118, 7.3733330, -38.6027641, 7.3181982, -44.1978836, 44.2058105
7: -15.1989155, 30.6904221, -15.1779509, 30.5581856, -41.5735550, 41.6898651
8: -15.2828693, 34.5627174, -15.2687883, 34.4539070, -46.8278198, 46.9331131
9: -10.2807331, 27.0785828, -10.2514153, 27.1042881, -35.6331787, 35.6062851
10: -28.4067726, 23.7008667, -28.3733330, 23.7846622, -50.3119507, 50.2156219
11: -35.7250214, 14.0577421, -35.7080917, 14.1098919, -49.1791382, 49.0907593
12: -49.4782104, 1.7071810, -49.3271561, 1.7924142, -43.9473190, 43.7029648
13: -28.7594585, 21.2811279, -28.7202530, 21.2413864, -49.3874054, 49.3983612
14: -70.9690018, -6.5703049, -70.8226242, -6.5506210, -64.4183807, 64.2523193
15: -17.2873535, 24.6327190, -17.2987862, 24.6305981, -41.9179535, 41.9315033
16: -27.4025631, 23.5238800, -27.4239635, 23.5801277, -48.2761078, 48.2556992
17: -71.2095490, -4.0136185, -71.0909729, -3.9981918, -67.2113571, 67.0773544
18: -34.8143463, 11.6408377, -34.6936111, 11.6063690, -40.5639839, 40.4768219
19: -25.7017288, 5.1959467, -25.6406097, 5.1941509, -29.8516693, 29.7903786
20: -26.4659157, 4.2601681, -26.4210300, 4.2683945, -29.0229073, 28.9541283
21: -31.3600426, 9.9931278, -31.2885914, 10.0056906, -40.2548447, 40.1711044
22: -33.6220360, 6.8883877, -33.5409927, 6.8575363, -38.3700027, 38.3182373
23: -26.9084206, 8.8143806, -26.8527184, 8.8050842, -35.1836243, 35.1255875
24: -23.2191963, 9.8290367, -23.1785259, 9.7952490, -32.6924744, 32.6737595
25: -29.2061653, 6.0161171, -29.1489353, 5.9933205, -34.2684021, 34.2248611
26: -43.0879364, 7.6632042, -42.8623886, 7.6803474, -43.7305450, 43.4925537
27: -26.6074486, 11.4429674, -26.5420074, 11.4083929, -37.6411667, 37.6088638
28: -29.6116772, 7.1187868, -29.5281067, 7.0789585, -36.5303574, 36.4948273
29: -32.6189804, 8.8993301, -32.5332718, 8.8740120, -41.4929924, 41.4326019
30: -37.6019821, 6.8448534, -37.5179520, 6.8192739, -44.4212570, 44.3628044
31: -31.4058552, 7.2374363, -31.3431778, 7.2203674, -37.5533752, 37.4919891
32: -33.6394806, 6.6294031, -33.5877380, 6.6310844, -40.2705650, 40.2171402
33: -43.8174553, 15.9983435, -43.8416901, 15.9605131, -56.9735870, 57.0557556
34: -50.6299095, -4.2800074, -50.6055069, -4.3364034, -42.1451263, 42.1886978
35: -40.8552475, 7.0519023, -40.8640366, 7.0135574, -43.7421494, 43.7987213
36: -44.4099503, 5.3748703, -44.3459282, 5.3398709, -45.3980484, 45.3738861
37: -59.4127312, 2.3402371, -59.3292198, 2.3193383, -55.0763702, 55.0240707
38: -50.8493958, 8.5621662, -50.7958603, 8.5504971, -59.3998947, 59.3580246
39: -52.1070328, 14.8606529, -52.0563049, 14.8670759, -66.9741058, 66.9169617
40: -47.7876434, 8.3137760, -47.7598801, 8.2801819, -53.0282059, 53.0393677
41: -31.7901859, 15.1876736, -31.7555008, 15.1508484, -45.3337402, 45.3217010
42: -27.1152000, 10.0211058, -27.0994148, 10.0392933, -36.6061096, 36.5396118

Time for backsubstitution: 1.94 seconds

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
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6972172, upper bound: 14.7752221
time: 49.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7420616, upper bound: 14.7752221
time: 36.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.7652416, 32.9412613, -23.7956257, 32.8197289, -54.1560211, 54.3064651
1: -7.5995312, 32.3722534, -7.6342525, 32.3033981, -36.4067612, 36.5087051
2: -4.6249952, 31.8412361, -4.7071333, 31.7562809, -33.0105743, 33.1865044
3: -8.8339157, 29.0033646, -8.8848629, 28.8660946, -32.3186111, 32.5219269
4: -9.9614964, 35.1053886, -10.0494366, 35.0290565, -43.2303162, 43.3980255
5: -11.0052567, 29.9982891, -11.0710087, 29.8685074, -37.9816513, 38.1957703
6: -38.6908302, 7.3747072, -38.6906700, 7.4523048, -44.3590317, 44.2901382
7: -15.2005138, 30.7015839, -15.2382698, 30.5930767, -41.6089325, 41.7618332
8: -15.2848549, 34.5779190, -15.3508644, 34.5038567, -46.8765411, 47.0307312
9: -10.2781582, 27.0804901, -10.2802773, 27.1383209, -35.6671982, 35.6418076
10: -28.4088821, 23.7122803, -28.4524918, 23.8365097, -50.3654633, 50.3470383
11: -35.7264290, 14.0673018, -35.7557297, 14.1496477, -49.2212524, 49.1468124
12: -49.4881592, 1.7090273, -49.3656311, 1.8363738, -44.0079803, 43.7469635
13: -28.7624645, 21.2815876, -28.7616482, 21.2689152, -49.4150238, 49.4501495
14: -70.9711456, -6.5449371, -70.9638138, -6.4752369, -64.4959106, 64.4188766
15: -17.2887535, 24.6348667, -17.3325253, 24.6472702, -41.9360237, 41.9673920
16: -27.4037304, 23.5269966, -27.4816170, 23.6066170, -48.3048630, 48.3175659
17: -71.2144165, -3.9735508, -71.2819672, -3.8851585, -67.3292542, 67.3084183
18: -34.8159866, 11.6393261, -34.7365570, 11.6300354, -40.5944977, 40.5157661
19: -25.7032928, 5.1990948, -25.6740723, 5.2172365, -29.8748703, 29.8212128
20: -26.4660797, 4.2614183, -26.4314938, 4.2915173, -29.0447159, 28.9692230
21: -31.3609161, 9.9958725, -31.3189621, 10.0290985, -40.2787628, 40.1989212
22: -33.6240387, 6.8912101, -33.5760269, 6.8826880, -38.3958817, 38.3756790
23: -26.9096584, 8.8166924, -26.8792515, 8.8293362, -35.2100296, 35.1535683
24: -23.2203445, 9.8347206, -23.2353210, 9.8314400, -32.7285805, 32.7343674
25: -29.2064838, 6.0208797, -29.1978550, 6.0281157, -34.3053436, 34.2847977
26: -43.0896263, 7.6625147, -42.8881226, 7.6934857, -43.7512360, 43.5237656
27: -26.6162052, 11.4438810, -26.5881424, 11.4723721, -37.7343063, 37.6592636
28: -29.6124001, 7.1209192, -29.5465984, 7.1214399, -36.5788879, 36.5156479
29: -32.6200142, 8.9022503, -32.5643997, 8.8988018, -41.5188141, 41.4666519
30: -37.6018372, 6.8475771, -37.5324707, 6.8478222, -44.4496613, 44.3800468
31: -31.4082184, 7.2473669, -31.3818817, 7.2607884, -37.5900764, 37.5316010
32: -33.6666985, 6.6301775, -33.6780815, 6.7679977, -40.4346962, 40.3082581
33: -43.8200417, 15.9997072, -43.8657837, 16.0042686, -57.0339966, 57.0885849
34: -50.6468582, -4.2791986, -50.6674004, -4.2457418, -42.2612915, 42.2513123
35: -40.8638611, 7.0526657, -40.9014244, 7.0612059, -43.8058243, 43.8346786
36: -44.4376907, 5.3756218, -44.4337616, 5.4627986, -45.5488968, 45.4616547
37: -59.4237061, 2.3418365, -59.3760490, 2.4024363, -55.1810608, 55.0735474
38: -50.8702011, 8.5635138, -50.8769226, 8.6553898, -59.5255890, 59.4404373
39: -52.1147614, 14.8615932, -52.1006432, 14.9151773, -67.0299377, 66.9622345
40: -47.8096390, 8.3151321, -47.8319702, 8.3985548, -53.1687012, 53.1110001
41: -31.8153038, 15.1887608, -31.8416901, 15.2841444, -45.4931793, 45.4070282
42: -27.1328773, 10.0224953, -27.1619968, 10.1395912, -36.7239609, 36.6003075

Time for backsubstitution: 1.92 seconds

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
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 772
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 62.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7711206, upper bound: 14.7752221
time: 41.02 seconds

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

Time for backsubstitution: 1.94 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 42.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 14.85 seconds

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

Time for backsubstitution: 1.93 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6951250, upper bound: 14.8130237
time: 38.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 34.44 seconds

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

Time for backsubstitution: 1.93 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6972172, upper bound: 14.8130237
time: 42.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7420616, upper bound: 14.8130237
time: 32.81 seconds

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

Time for backsubstitution: 1.81 seconds

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
time: 34.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7711206, upper bound: 14.8130237
time: 36.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -23.8176441, 32.8770409, -23.7691650, 32.8059502, -54.1607208, 54.2108154
1: -7.6353378, 32.3251572, -7.6087141, 32.2778015, -36.4285889, 36.4274025
2: -4.6866884, 31.7778511, -4.6635451, 31.7103252, -32.9716949, 33.0768280
3: -8.8736763, 28.9047871, -8.8790359, 28.8606377, -32.3002090, 32.3966217
4: -10.0525646, 35.0611038, -10.0155735, 34.9859161, -43.2389221, 43.3002014
5: -11.0778265, 29.9175739, -11.0848608, 29.8616486, -37.9863281, 38.1188126
6: -38.6989861, 7.3974009, -38.6300850, 7.3369198, -44.3147430, 44.2831497
7: -15.2274160, 30.6211395, -15.2107468, 30.5672798, -41.5937881, 41.6481628
8: -15.3776474, 34.5546913, -15.3317432, 34.4645500, -46.8751984, 46.9730072
9: -10.3052130, 27.1856155, -10.2565670, 27.1702290, -35.7342911, 35.6250877
10: -28.4230022, 23.9145393, -28.3829441, 23.9197083, -50.4649658, 50.2716293
11: -35.7741203, 14.2352324, -35.7351418, 14.2174988, -49.3269806, 49.2289505
12: -49.3974686, 1.9016838, -49.3394203, 1.9275508, -43.9991989, 43.8469849
13: -28.7711926, 21.3144875, -28.7314072, 21.2892494, -49.4581680, 49.4515762
14: -70.9171982, -6.4271507, -70.8341217, -6.4592113, -64.4579849, 64.4069672
15: -17.3798828, 24.6745548, -17.3550873, 24.6517887, -42.0316696, 42.0296402
16: -27.4826431, 23.6635017, -27.4440784, 23.6571865, -48.4441986, 48.3319473
17: -71.1351624, -3.8868446, -71.1043549, -3.9114609, -67.2237015, 67.2175140
18: -34.7379265, 11.6289387, -34.7115173, 11.6126842, -40.4981194, 40.4948235
19: -25.6909523, 5.2334042, -25.6584682, 5.2201309, -29.8429832, 29.8427963
20: -26.4653473, 4.3242927, -26.4366074, 4.3099389, -29.0303574, 28.9687347
21: -31.3477287, 10.0811491, -31.3069534, 10.0644245, -40.3098602, 40.1745300
22: -33.6023254, 6.9139781, -33.5611496, 6.8861980, -38.3696213, 38.3575287
23: -26.9001541, 8.8589325, -26.8640671, 8.8341322, -35.1572418, 35.1496887
24: -23.2454109, 9.8318100, -23.2113762, 9.8009567, -32.7356911, 32.7502098
25: -29.2034588, 6.0573840, -29.1650791, 6.0222659, -34.2683182, 34.2959366
26: -42.9391022, 7.7201433, -42.8853035, 7.7429366, -43.6602020, 43.5303497
27: -26.6226845, 11.4392405, -26.5753574, 11.4130421, -37.6751251, 37.6482506
28: -29.5896149, 7.1260781, -29.5416527, 7.0883517, -36.5215302, 36.5397720
29: -32.5938988, 8.9399967, -32.5443268, 8.9062309, -41.5001297, 41.4843216
30: -37.5820770, 6.9060135, -37.5356216, 6.8596649, -44.4417419, 44.4416351
31: -31.3956757, 7.2611427, -31.3638248, 7.2377362, -37.5211220, 37.6044159
32: -33.6739807, 6.7141857, -33.6011658, 6.6826000, -40.3565826, 40.3153534
33: -43.9360924, 15.9977264, -43.9153061, 15.9758873, -57.0551529, 57.1142883
34: -50.6991463, -4.2882509, -50.6484489, -4.3297291, -42.1879578, 42.2201385
35: -40.9571342, 7.0500360, -40.9253235, 7.0224562, -43.8064194, 43.8599701
36: -44.4449615, 5.3890429, -44.3669319, 5.3560009, -45.4570770, 45.4091415
37: -59.4264069, 2.3556185, -59.3602180, 2.3377099, -55.1263657, 55.0885391
38: -50.8960037, 8.5940170, -50.8315887, 8.5782757, -59.4742813, 59.4256058
39: -52.1416016, 14.8960114, -52.0858917, 14.8993549, -67.0409546, 66.9819031
40: -47.8425713, 8.3119001, -47.7976265, 8.2899313, -53.0884857, 53.0714645
41: -31.8549366, 15.2191925, -31.7839851, 15.1763687, -45.4152069, 45.3817215
42: -27.1655560, 10.1406689, -27.1155052, 10.1119518, -36.7293701, 36.6917801

Time for backsubstitution: 1.84 seconds

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
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
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
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 546
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 42.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 35.17 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.8198318, 32.8843155, -23.8427238, 32.8359642, -54.1940842, 54.2915344
1: -7.6362963, 32.3332481, -7.6598921, 32.3054543, -36.4535370, 36.4882545
2: -4.6885715, 31.7937679, -4.7561779, 31.7590370, -33.0198669, 33.1853027
3: -8.8755665, 28.9071617, -8.9270639, 28.8770084, -32.3160400, 32.4500923
4: -10.0542307, 35.0764427, -10.1076603, 35.0358734, -43.2894592, 43.4064713
5: -11.0797005, 29.9217987, -11.1310234, 29.8841038, -38.0086212, 38.1782684
6: -38.7265816, 7.3988113, -38.7179489, 7.4709892, -44.4759216, 44.3674164
7: -15.2289848, 30.6323681, -15.2710743, 30.6022358, -41.6292114, 41.7201233
8: -15.3795967, 34.5699768, -15.4138336, 34.5144806, -46.9239349, 47.0706253
9: -10.3026152, 27.1875381, -10.2854347, 27.2041988, -35.7683105, 35.6605759
10: -28.4250698, 23.9259586, -28.4621296, 23.9715309, -50.5185699, 50.4030075
11: -35.7755165, 14.2447710, -35.7826881, 14.2572479, -49.3691635, 49.2850800
12: -49.4074020, 1.9035234, -49.3779297, 1.9714608, -44.0598068, 43.8909607
13: -28.7741928, 21.3149376, -28.7728271, 21.3167686, -49.4857407, 49.5033646
14: -70.9192657, -6.4017525, -70.9752731, -6.3837814, -64.5354843, 64.5735168
15: -17.3813057, 24.6767082, -17.3888359, 24.6684513, -42.0497589, 42.0655441
16: -27.4838409, 23.6666012, -27.5016670, 23.6837978, -48.4729614, 48.3938370
17: -71.1400528, -3.8467617, -71.2952194, -3.7983112, -67.3417435, 67.4484558
18: -34.7395668, 11.6274147, -34.7544212, 11.6363640, -40.5285721, 40.5337410
19: -25.6925411, 5.2365360, -25.6919155, 5.2432299, -29.8661880, 29.8735580
20: -26.4655075, 4.3255301, -26.4470596, 4.3330064, -29.0521889, 28.9837875
21: -31.3485603, 10.0838890, -31.3373013, 10.0878086, -40.3337860, 40.2023392
22: -33.6042671, 6.9167571, -33.5961761, 6.9113560, -38.3955307, 38.4149551
23: -26.9013786, 8.8612576, -26.8906174, 8.8584080, -35.1836395, 35.1776237
24: -23.2465782, 9.8375072, -23.2680855, 9.8371086, -32.7717934, 32.8107758
25: -29.2038326, 6.0621715, -29.2140045, 6.0571122, -34.3053131, 34.3558044
26: -42.9407806, 7.7194538, -42.9110374, 7.7560687, -43.6809540, 43.5615082
27: -26.6314735, 11.4401264, -26.6215515, 11.4770088, -37.7683258, 37.6987762
28: -29.5903778, 7.1282272, -29.5602150, 7.1308374, -36.5700455, 36.5606461
29: -32.5948677, 8.9429417, -32.5754204, 8.9309855, -41.5258522, 41.5183640
30: -37.5819778, 6.9087296, -37.5501289, 6.8883152, -44.4702911, 44.4588585
31: -31.3980637, 7.2710485, -31.4024982, 7.2781401, -37.5578423, 37.6439438
32: -33.7012138, 6.7149839, -33.6915054, 6.8194313, -40.5206451, 40.4064903
33: -43.9387245, 15.9991121, -43.9395027, 16.0196152, -57.1157684, 57.1471405
34: -50.7161331, -4.2874546, -50.7103653, -4.2390575, -42.3040848, 42.2828064
35: -40.9657822, 7.0507951, -40.9627571, 7.0701346, -43.8701096, 43.8959579
36: -44.4727707, 5.3898301, -44.4548569, 5.4789276, -45.6079330, 45.4969559
37: -59.4373436, 2.3572545, -59.4071045, 2.4207788, -55.2310486, 55.1379929
38: -50.9168472, 8.5954332, -50.9126587, 8.6832199, -59.6000671, 59.5080910
39: -52.1493530, 14.8970013, -52.1302605, 14.9474201, -67.0967712, 67.0272598
40: -47.8645859, 8.3132191, -47.8696594, 8.4082975, -53.2289734, 53.1431046
41: -31.8800278, 15.2203207, -31.8702202, 15.3096228, -45.5746460, 45.4670486
42: -27.1832180, 10.1420755, -27.1780300, 10.2122688, -36.8471909, 36.7524490

Time for backsubstitution: 1.85 seconds

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
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1543
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
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 546
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7204540, upper bound: 14.7752221
time: 36.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 38.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -23.9054852, 33.0286560, -23.7726116, 32.8063812, -54.2504578, 54.3748322
1: -7.6692114, 32.3855247, -7.6083651, 32.2775497, -36.4701462, 36.4874802
2: -4.7510471, 31.9108562, -4.6700239, 31.7098694, -33.0341568, 33.2233582
3: -8.9477568, 29.0834599, -8.8867321, 28.8601208, -32.3659973, 32.5822754
4: -10.0984116, 35.1694031, -10.0187502, 34.9854355, -43.2869415, 43.4163742
5: -11.1609955, 30.1102123, -11.0941505, 29.8614349, -38.0660629, 38.3227844
6: -38.7539520, 7.4162655, -38.6300087, 7.3336411, -44.4138947, 44.3014374
7: -15.3018608, 30.7218475, -15.2131424, 30.5660324, -41.6646347, 41.7585526
8: -15.4300327, 34.6764679, -15.3343658, 34.4643097, -46.9283371, 47.1131897
9: -10.3682775, 27.2229118, -10.2556505, 27.1688423, -35.7930298, 35.6675377
10: -28.6379662, 24.0001068, -28.3830376, 23.9267101, -50.6927109, 50.3553619
11: -35.9979439, 14.2879829, -35.7349472, 14.2224407, -49.5686798, 49.2837601
12: -49.6912613, 2.0304112, -49.3379860, 1.9392200, -44.3059387, 43.9648056
13: -28.7946892, 21.4107208, -28.7275715, 21.2906723, -49.4816437, 49.5393219
14: -71.1285553, -6.3501167, -70.8330383, -6.4497833, -64.6787720, 64.4829254
15: -17.4036198, 24.7702808, -17.3478451, 24.6519527, -42.0555725, 42.1181259
16: -27.5833359, 23.6759872, -27.4434433, 23.6535988, -48.5148926, 48.3428497
17: -71.2856598, -3.8015976, -71.1033478, -3.9016800, -67.3839798, 67.3017502
18: -34.9036560, 11.6604300, -34.7112732, 11.6126957, -40.6773376, 40.5404663
19: -25.7982521, 5.2543268, -25.6588459, 5.2222805, -29.9351425, 29.8588142
20: -26.5760021, 4.3559585, -26.4371490, 4.3122225, -29.1581001, 29.0031013
21: -31.5228767, 10.1240120, -31.3069878, 10.0691223, -40.5013123, 40.2202530
22: -33.6779633, 6.9593735, -33.5598145, 6.8836846, -38.4508362, 38.4017410
23: -27.0093708, 8.8840218, -26.8641548, 8.8355370, -35.2570419, 35.1796494
24: -23.3058624, 9.8471088, -23.2114544, 9.7986746, -32.7881432, 32.7677345
25: -29.2723999, 6.0910497, -29.1646500, 6.0241299, -34.3283539, 34.3300323
26: -43.2224350, 7.8355398, -42.8841591, 7.7533388, -43.9565506, 43.6356430
27: -26.6983280, 11.4569111, -26.5760574, 11.4120731, -37.7448883, 37.6671829
28: -29.6560287, 7.1523118, -29.5419788, 7.0875659, -36.5764236, 36.5810852
29: -32.6708298, 8.9779205, -32.5430870, 8.9080429, -41.5788727, 41.5210075
30: -37.7004776, 6.9469471, -37.5355377, 6.8612709, -44.5617485, 44.4824829
31: -31.5304661, 7.2770495, -31.3644867, 7.2387810, -37.6231537, 37.6327896
32: -33.7543716, 6.7432051, -33.6007996, 6.6818380, -40.4362106, 40.3440056
33: -43.9902534, 16.1111755, -43.9193535, 15.9756441, -57.1090317, 57.2457504
34: -50.7314148, -4.2056522, -50.6505699, -4.3298922, -42.2174759, 42.3057098
35: -40.9945450, 7.1511350, -40.9281273, 7.0226717, -43.8431702, 43.9626236
36: -44.4619560, 5.4276481, -44.3663330, 5.3542376, -45.4761505, 45.4520187
37: -59.5022278, 2.3875465, -59.3599319, 2.3335695, -55.1958160, 55.1398773
38: -50.9421043, 8.6361704, -50.8310547, 8.5760031, -59.5181084, 59.4672241
39: -52.1929321, 14.9469500, -52.0862808, 14.8955412, -67.0884705, 67.0332336
40: -47.8951492, 8.3513165, -47.7981148, 8.2882605, -53.1423798, 53.1140442
41: -31.9099960, 15.2420340, -31.7840614, 15.1733665, -45.4849701, 45.4017105
42: -27.2510986, 10.1715260, -27.1153793, 10.1070261, -36.8703232, 36.7294540

Time for backsubstitution: 1.82 seconds

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
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1584
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
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7391718, upper bound: 14.7752221
time: 33.28 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7840025, upper bound: 14.7752221
time: 229.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.9076729, 33.0358849, -23.8461609, 32.8364410, -54.2838287, 54.4555511
1: -7.6701927, 32.3935623, -7.6595535, 32.3051910, -36.4951324, 36.5483398
2: -4.7529173, 31.9267578, -4.7626486, 31.7586613, -33.0823135, 33.3317642
3: -8.9496164, 29.0858593, -8.9347935, 28.8764629, -32.3818283, 32.6357079
4: -10.1000443, 35.1847076, -10.1107922, 35.0354691, -43.3375168, 43.5226440
5: -11.1628599, 30.1144295, -11.1403036, 29.8838863, -38.0883331, 38.3822174
6: -38.7815285, 7.4176760, -38.7178726, 7.4676743, -44.5750198, 44.3856735
7: -15.3034391, 30.7330093, -15.2735138, 30.6009197, -41.7000961, 41.8305283
8: -15.4320164, 34.6916733, -15.4164219, 34.5142441, -46.9770432, 47.2107773
9: -10.3657179, 27.2248383, -10.2845039, 27.2028637, -35.8270264, 35.7030258
10: -28.6400909, 24.0115089, -28.4622002, 23.9785442, -50.7462616, 50.4867096
11: -35.9993591, 14.2975197, -35.7825356, 14.2622089, -49.6108398, 49.3397598
12: -49.7011833, 2.0322175, -49.3764534, 1.9831171, -44.3665619, 44.0088120
13: -28.7976875, 21.4111443, -28.7689819, 21.3182163, -49.5092392, 49.5911636
14: -71.1306458, -6.3247948, -70.9741440, -6.3744087, -64.7562408, 64.6493530
15: -17.4050350, 24.7724133, -17.3815765, 24.6686172, -42.0736542, 42.1539917
16: -27.5845242, 23.6791286, -27.5010777, 23.6800728, -48.5436859, 48.4047012
17: -71.2905197, -3.7614784, -71.2942047, -3.7884293, -67.5020905, 67.5327301
18: -34.9052963, 11.6589012, -34.7541771, 11.6363897, -40.7078171, 40.5793648
19: -25.7998219, 5.2574501, -25.6922817, 5.2453775, -29.9584122, 29.8895874
20: -26.5761509, 4.3572211, -26.4476013, 4.3353014, -29.1799240, 29.0181503
21: -31.5237045, 10.1267319, -31.3373203, 10.0925236, -40.5252380, 40.2480545
22: -33.6799011, 6.9621763, -33.5948486, 6.9088597, -38.4767456, 38.4591675
23: -27.0105820, 8.8863182, -26.8906918, 8.8598347, -35.2834778, 35.2076225
24: -23.3070011, 9.8527794, -23.2681656, 9.8348408, -32.8242836, 32.8283081
25: -29.2727699, 6.0958381, -29.2135544, 6.0589981, -34.3653336, 34.3899422
26: -43.2240982, 7.8348932, -42.9099770, 7.7664318, -43.9773102, 43.6667328
27: -26.7070656, 11.4578171, -26.6222610, 11.4760294, -37.8380280, 37.7176895
28: -29.6567764, 7.1544681, -29.5605545, 7.1300716, -36.6249771, 36.6019592
29: -32.6718826, 8.9808054, -32.5742111, 8.9328337, -41.6047173, 41.5550156
30: -37.7003822, 6.9496975, -37.5500526, 6.8898849, -44.5902672, 44.4997482
31: -31.5328293, 7.2869868, -31.4031525, 7.2792296, -37.6598701, 37.6723061
32: -33.7815819, 6.7439547, -33.6911507, 6.8187037, -40.6002846, 40.4351044
33: -43.9928207, 16.1125717, -43.9434776, 16.0193863, -57.1696472, 57.2786560
34: -50.7484131, -4.2048521, -50.7124863, -4.2392125, -42.3336258, 42.3683624
35: -41.0031586, 7.1518393, -40.9655113, 7.0703363, -43.9068146, 43.9986191
36: -44.4897041, 5.4284010, -44.4542313, 5.4772024, -45.6270370, 45.5398178
37: -59.5131645, 2.3892226, -59.4068565, 2.4166908, -55.3004761, 55.1893005
38: -50.9628296, 8.6375332, -50.9121361, 8.6809330, -59.6437607, 59.5496674
39: -52.2006226, 14.9479513, -52.1306839, 14.9436207, -67.1442413, 67.0786362
40: -47.9170990, 8.3526230, -47.8701820, 8.4066296, -53.2828522, 53.1856842
41: -31.9350548, 15.2430649, -31.8703232, 15.3066492, -45.6444397, 45.4870987
42: -27.2687302, 10.1729355, -27.1779175, 10.2073193, -36.9881668, 36.7901306

Time for backsubstitution: 1.88 seconds

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
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1461
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1584
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
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7682095, upper bound: 14.7752221
time: 35.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.8130231, upper bound: 14.7752221
time: 41.54 seconds

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

Time for backsubstitution: 1.89 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237
time: 40.04 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 28.73 seconds

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

Time for backsubstitution: 1.75 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7204540, upper bound: 14.8130237
time: 46.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 35.85 seconds

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

Time for backsubstitution: 1.74 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7391718, upper bound: 14.8130237
time: 40.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 43.19 seconds

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

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7682095, upper bound: 14.8130237
time: 41.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237
time: 34.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 78.30 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7399635, upper bound: 14.7752221
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6972172, upper bound: 14.7752221
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7420616, upper bound: 14.7752221
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7711206, upper bound: 14.7752221
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
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7204540, upper bound: 14.7752221
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7391718, upper bound: 14.7752221
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7840025, upper bound: 14.7752221
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.7682095, upper bound: 14.7752221
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 78.30
Output dim: 2, lower bound: -14.8130231, upper bound: 14.7752221
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

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -23.4591312, 32.7256622, -23.6248512, 32.7858276, -53.8139801, 54.6640778
1: -7.4229717, 32.2615852, -7.5218277, 32.2734489, -35.9734039, 36.2833977
2: -4.4266758, 31.6584625, -4.5495300, 31.7046566, -32.6678543, 32.8321991
3: -8.6589289, 28.7898426, -8.7854681, 28.8458252, -31.7532654, 32.2047653
4: -9.7315865, 34.9334602, -9.8762236, 34.9761314, -42.9472122, 43.1888580
5: -10.8055201, 29.7586098, -10.9651489, 29.8405380, -37.7599182, 37.9944000
6: -38.5659485, 7.2521696, -38.5961723, 7.2781773, -43.2866135, 44.0888290
7: -14.9365778, 30.5423088, -15.0935221, 30.5571136, -40.1161728, 41.4382629
8: -15.0525560, 34.3900909, -15.1885767, 34.4484634, -46.8808899, 46.6572571
9: -10.0752611, 26.9963799, -10.1909561, 27.1016102, -35.4259186, 35.8345642
10: -28.0854931, 23.5584373, -28.3274326, 23.7682457, -49.9364777, 50.0282822
11: -35.4684563, 13.9640360, -35.6944122, 14.0941992, -49.4025803, 48.9795456
12: -49.1458969, 1.4080563, -49.3236504, 1.7064209, -43.3936081, 43.4061050
13: -28.6979160, 21.1456318, -28.7083702, 21.2276077, -49.3097992, 49.8540039
14: -70.6236191, -6.6951485, -70.7615509, -6.5657921, -64.0578308, 64.0664062
15: -17.1935863, 24.5177231, -17.2775497, 24.6252403, -41.8188248, 41.7952728
16: -27.1468735, 23.4621735, -27.3555775, 23.5784512, -47.5068665, 48.1218185
17: -71.0115509, -4.1272125, -71.0712051, -4.0139751, -66.9975739, 66.9439926
18: -34.6259613, 11.5787983, -34.6859283, 11.5941124, -40.3625603, 40.2165985
19: -25.5629005, 5.1623087, -25.6297607, 5.1861715, -29.7202034, 29.5366707
20: -26.3307858, 4.2123094, -26.4136047, 4.2584147, -29.2358475, 28.9022179
21: -31.1408787, 9.9312363, -31.2735176, 9.9935379, -40.1414261, 40.1001053
22: -33.4957886, 6.7846894, -33.5304413, 6.8340459, -38.2094193, 36.5863037
23: -26.7710209, 8.7516460, -26.8448086, 8.7883186, -35.0411758, 34.7310753
24: -23.1285610, 9.7953968, -23.1693707, 9.7884064, -32.6053619, 32.3657036
25: -29.0963593, 5.9054737, -29.1403484, 5.9570689, -34.1389999, 32.6977921
26: -42.7666702, 7.5071521, -42.8540459, 7.6502156, -43.5991821, 43.3308182
27: -26.4987793, 11.3883972, -26.5305862, 11.3983393, -37.8971176, 37.5363541
28: -29.5083733, 7.0229621, -29.5201416, 7.0496087, -36.4078445, 36.2445450
29: -32.5027161, 8.8020716, -32.5237656, 8.8458939, -41.3486099, 41.3258362
30: -37.4684105, 6.7682886, -37.5092926, 6.8039160, -44.2723274, 44.2775803
31: -31.2265663, 7.1907001, -31.3308506, 7.2067389, -37.3945694, 36.9776344
32: -33.5170441, 6.5083590, -33.5807838, 6.5916815, -39.8436050, 40.0891418
33: -43.6859970, 15.7949219, -43.8198395, 15.9207115, -56.8007660, 56.8674088
34: -50.5468559, -4.4755955, -50.5966454, -4.3855591, -42.0018768, 41.9521942
35: -40.7586021, 6.8389139, -40.8515930, 6.9642544, -43.5842972, 43.6719208
36: -44.3371239, 5.2088480, -44.3391571, 5.2859726, -45.2585373, 45.2779541
37: -59.2691078, 2.2413836, -59.3139267, 2.2942486, -54.8976822, 55.2941742
38: -50.7287369, 8.4168720, -50.7851791, 8.5083361, -59.2370720, 59.2020493
39: -51.9771652, 14.7862701, -52.0349350, 14.8610029, -66.8381653, 66.8212051
40: -47.6762505, 8.2409096, -47.7451973, 8.2680216, -52.8274536, 52.9499359
41: -31.6884289, 15.0892172, -31.7447281, 15.1215649, -44.8299561, 45.2180176
42: -27.0033722, 9.9072323, -27.0937748, 10.0061665, -36.3753586, 36.4148636

Time for backsubstitution: 1.83 seconds

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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1572
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
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 878
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1675
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
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 772
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
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7541369
time: 41.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 29.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -23.6730118, 32.7833519, -23.7179470, 32.7891197, -54.0263214, 54.8323059
1: -7.5621300, 32.3037605, -7.5829220, 32.2759399, -36.1014862, 36.3915710
2: -4.5594034, 31.6961174, -4.6080732, 31.7079601, -32.7882538, 32.9333191
3: -8.7586937, 28.8238125, -8.8289022, 28.8502426, -31.8312683, 32.2903214
4: -9.9095354, 34.9819984, -9.9535646, 34.9795456, -43.1154938, 43.3154984
5: -10.9207077, 29.8040524, -11.0154238, 29.8462334, -37.8632736, 38.0973358
6: -38.6089439, 7.3526993, -38.6028137, 7.3211384, -43.4161758, 44.1854324
7: -15.1223946, 30.5906239, -15.1751623, 30.5594559, -40.2553558, 41.5804443
8: -15.2306747, 34.4456253, -15.2659798, 34.4540672, -47.0431366, 46.7978058
9: -10.2152338, 27.0407104, -10.2518597, 27.1056385, -35.5494843, 35.9431381
10: -28.1905975, 23.6150360, -28.3728809, 23.7776299, -50.0415649, 50.1312332
11: -35.5011139, 14.0044155, -35.7081337, 14.1048641, -49.4785156, 49.0387573
12: -49.1978683, 1.5819793, -49.3285599, 1.7801714, -43.5336151, 43.5313797
13: -28.7358894, 21.1776199, -28.7240448, 21.2399292, -49.3604660, 49.9016647
14: -70.7631531, -6.6407032, -70.8233185, -6.5578518, -64.2052994, 64.1826172
15: -17.2681961, 24.5488586, -17.3081818, 24.6303482, -41.8985443, 41.8570404
16: -27.3040085, 23.5129967, -27.4241600, 23.5837498, -47.6419449, 48.2449951
17: -71.0611496, -4.0938110, -71.0918579, -4.0073967, -67.0537567, 66.9980469
18: -34.6509705, 11.6090603, -34.6938171, 11.6062536, -40.3896179, 40.2693329
19: -25.5970497, 5.1760454, -25.6401501, 5.1919656, -29.7703514, 29.5611954
20: -26.3557377, 4.2286658, -26.4204445, 4.2661099, -29.3192062, 28.9191017
21: -31.1850452, 9.9499607, -31.2884998, 10.0009022, -40.2098999, 40.1268616
22: -33.5446205, 6.8435421, -33.5422058, 6.8598728, -38.2861328, 36.6176224
23: -26.8003082, 8.7878475, -26.8525772, 8.8033943, -35.0974960, 34.7775536
24: -23.1537037, 9.8131390, -23.1784821, 9.7975121, -32.6436005, 32.3928070
25: -29.1385002, 5.9823613, -29.1494217, 5.9911304, -34.2173996, 32.7661438
26: -42.8111267, 7.5548773, -42.8634529, 7.6710033, -43.6661453, 43.3738937
27: -26.5293541, 11.4252567, -26.5412178, 11.4093208, -37.9386749, 37.5887756
28: -29.5456581, 7.0915051, -29.5276775, 7.0794964, -36.4769516, 36.3202362
29: -32.5434570, 8.8615627, -32.5344009, 8.8720284, -41.4154854, 41.3959656
30: -37.4902878, 6.8049707, -37.5180130, 6.8175993, -44.3078880, 44.3229828
31: -31.2720947, 7.2208862, -31.3423901, 7.2191668, -37.4827347, 37.0234718
32: -33.5592880, 6.5992031, -33.5880470, 6.6318541, -39.9585953, 40.1872482
33: -43.7628250, 15.8841114, -43.8375053, 15.9605780, -56.9190063, 56.9335480
34: -50.5976486, -4.3639336, -50.6033630, -4.3366494, -42.1157379, 42.0293732
35: -40.8177681, 6.9490776, -40.8611641, 7.0129318, -43.7049637, 43.7416000
36: -44.3927155, 5.3346190, -44.3464508, 5.3413687, -45.3781738, 45.3836670
37: -59.3371964, 2.3074412, -59.3293991, 2.3231854, -55.0056229, 55.3229523
38: -50.8034172, 8.5219498, -50.7963638, 8.5536356, -59.3570518, 59.3183136
39: -52.0551834, 14.8093872, -52.0557938, 14.8707161, -66.9259033, 66.8651810
40: -47.7347717, 8.2740307, -47.7591896, 8.2817450, -52.8980103, 52.9921646
41: -31.7368584, 15.1640692, -31.7552700, 15.1535587, -44.9219513, 45.2955246
42: -27.0390587, 9.9958363, -27.0996017, 10.0439892, -36.4892502, 36.4992599

Time for backsubstitution: 1.83 seconds

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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 546
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7541369
time: 28.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 33.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -23.4612961, 32.7328949, -23.6984386, 32.8158646, -53.8473358, 54.7433777
1: -7.4239006, 32.2696648, -7.5730324, 32.3010292, -35.9983521, 36.3442001
2: -4.4285021, 31.6743317, -4.6422014, 31.7534142, -32.7162399, 32.9406357
3: -8.6608295, 28.7921638, -8.8334475, 28.8622112, -31.7685089, 32.2581406
4: -9.7332401, 34.9488029, -9.9682941, 35.0261497, -42.9977417, 43.2950134
5: -10.8074274, 29.7628288, -11.0112753, 29.8629837, -37.7822037, 38.0518646
6: -38.5935402, 7.2535515, -38.6840401, 7.4122400, -43.4476166, 44.1731644
7: -14.9381189, 30.5535202, -15.1538353, 30.5920238, -40.1509705, 41.5101624
8: -15.0545387, 34.4053650, -15.2706356, 34.4983788, -46.9316177, 46.7548447
9: -10.0727024, 26.9982491, -10.2198362, 27.1356411, -35.4598923, 35.8685951
10: -28.0875931, 23.5698299, -28.4066696, 23.8200817, -49.9897614, 50.1596527
11: -35.4697571, 13.9735928, -35.7419815, 14.1339931, -49.4410858, 49.0356064
12: -49.1557693, 1.4099121, -49.3621140, 1.7503643, -43.4505081, 43.4501495
13: -28.7009048, 21.1459885, -28.7497940, 21.2551117, -49.3373871, 49.8957825
14: -70.6256943, -6.6697655, -70.9026794, -6.4904099, -64.1352844, 64.2329102
15: -17.1950092, 24.5198326, -17.3112717, 24.6418934, -41.8369026, 41.8311043
16: -27.1480064, 23.4652061, -27.4131908, 23.6051254, -47.5353546, 48.1836395
17: -71.0163651, -4.0871086, -71.2620850, -3.9008732, -67.1154938, 67.1749725
18: -34.6275787, 11.5772495, -34.7288666, 11.6178474, -40.3930435, 40.2540359
19: -25.5644531, 5.1654539, -25.6632156, 5.2092528, -29.7434006, 29.5663910
20: -26.3309078, 4.2135301, -26.4240685, 4.2814980, -29.2554474, 28.9172745
21: -31.1416969, 9.9339514, -31.3038616, 10.0169430, -40.1647797, 40.1278610
22: -33.4977112, 6.7875009, -33.5654793, 6.8591623, -38.2353592, 36.6260605
23: -26.7722664, 8.7539301, -26.8713417, 8.8126211, -35.0675583, 34.7592201
24: -23.1297035, 9.8010941, -23.2261429, 9.8246183, -32.6414833, 32.4247742
25: -29.0967121, 5.9102311, -29.1892624, 5.9918890, -34.1759262, 32.7517776
26: -42.7683449, 7.5064797, -42.8797760, 7.6633358, -43.6191177, 43.3619537
27: -26.5075932, 11.3892899, -26.5767422, 11.4623079, -37.9699020, 37.5867348
28: -29.5090904, 7.0251069, -29.5386524, 7.0920868, -36.4563141, 36.2657700
29: -32.5036850, 8.8050156, -32.5548744, 8.8706532, -41.3743362, 41.3598900
30: -37.4683266, 6.7709932, -37.5238228, 6.8325186, -44.3008461, 44.2948151
31: -31.2289238, 7.2006369, -31.3695393, 7.2472081, -37.4313622, 37.0157852
32: -33.5442581, 6.5091658, -33.6711617, 6.7284679, -40.0089493, 40.1803284
33: -43.6885529, 15.7962112, -43.8439598, 15.9644337, -56.8610687, 56.8965912
34: -50.5637817, -4.4747887, -50.6585083, -4.2948875, -42.1180725, 42.0151215
35: -40.7672195, 6.8396449, -40.8889885, 7.0118704, -43.6479111, 43.7064209
36: -44.3648300, 5.2095881, -44.4270325, 5.4089022, -45.4093781, 45.3657913
37: -59.2800751, 2.2429924, -59.3608131, 2.3773165, -55.0023651, 55.3418579
38: -50.7495537, 8.4182463, -50.8662796, 8.6133318, -59.3628845, 59.2845268
39: -51.9848442, 14.7871399, -52.0792847, 14.9090967, -66.8939438, 66.8664246
40: -47.6982346, 8.2422943, -47.8173141, 8.3864164, -52.9690933, 53.0215302
41: -31.7135258, 15.0902872, -31.8309441, 15.2548380, -44.9887848, 45.3033371
42: -27.0210171, 9.9086361, -27.1563416, 10.1065006, -36.4928894, 36.4756012

Time for backsubstitution: 1.75 seconds

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
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1738
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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7541368
time: 41.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
time: 47.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -23.6751957, 32.7905960, -23.7915344, 32.8191719, -54.0597229, 54.9116516
1: -7.5630627, 32.3118210, -7.6341543, 32.3035545, -36.1265182, 36.4524078
2: -4.5612812, 31.7120209, -4.7007265, 31.7567062, -32.8366470, 33.0417786
3: -8.7605801, 28.8261738, -8.8769054, 28.8665657, -31.8464584, 32.3436928
4: -9.9111795, 34.9973640, -10.0456209, 35.0295105, -43.1659775, 43.4216232
5: -10.9225969, 29.8082771, -11.0615206, 29.8686867, -37.8855972, 38.1548080
6: -38.6365700, 7.3540688, -38.6907005, 7.4552584, -43.5771713, 44.2697678
7: -15.1239395, 30.6018181, -15.2354622, 30.5943127, -40.2901611, 41.6523895
8: -15.2326279, 34.4608994, -15.3480129, 34.5039940, -47.0938721, 46.8954010
9: -10.2126436, 27.0426025, -10.2807102, 27.1396484, -35.5835190, 35.9771538
10: -28.1927052, 23.6264648, -28.4521084, 23.8294582, -50.0948105, 50.2626190
11: -35.5024452, 14.0139027, -35.7557411, 14.1445904, -49.5170746, 49.0948181
12: -49.2077827, 1.5838275, -49.3669777, 1.8240933, -43.5905228, 43.5753555
13: -28.7388630, 21.1780262, -28.7654762, 21.2674313, -49.3880615, 49.9435043
14: -70.7652206, -6.6153603, -70.9645157, -6.4824085, -64.2828140, 64.3491516
15: -17.2696056, 24.5509605, -17.3419266, 24.6469975, -41.9166031, 41.8928871
16: -27.3051720, 23.5160675, -27.4817848, 23.6103745, -47.6704178, 48.3068619
17: -71.0661011, -4.0536766, -71.2827988, -3.8943157, -67.1717834, 67.2291260
18: -34.6526031, 11.6075382, -34.7367210, 11.6299343, -40.4201431, 40.3067856
19: -25.5986061, 5.1791468, -25.6736336, 5.2150497, -29.7935486, 29.5909424
20: -26.3558674, 4.2298803, -26.4308853, 4.2891989, -29.3388367, 28.9341888
21: -31.1858616, 9.9526920, -31.3188515, 10.0243177, -40.2332687, 40.1546478
22: -33.5465698, 6.8463316, -33.5772552, 6.8850055, -38.3120651, 36.6574478
23: -26.8015614, 8.7901421, -26.8791237, 8.8276958, -35.1238785, 34.8057175
24: -23.1548634, 9.8188343, -23.2352600, 9.8336964, -32.6797295, 32.4518967
25: -29.1388531, 5.9871325, -29.1983280, 6.0259442, -34.2543488, 32.8201561
26: -42.8128052, 7.5542159, -42.8892670, 7.6840982, -43.6860809, 43.4050217
27: -26.5381260, 11.4261379, -26.5873337, 11.4732847, -38.0114098, 37.6391945
28: -29.5463829, 7.0936403, -29.5462055, 7.1219831, -36.5253983, 36.3414688
29: -32.5444717, 8.8644485, -32.5655174, 8.8967991, -41.4412689, 41.4299660
30: -37.4901276, 6.8077574, -37.5325470, 6.8461580, -44.3362846, 44.3403053
31: -31.2744465, 7.2308259, -31.3810539, 7.2595887, -37.5195007, 37.0615768
32: -33.5865402, 6.5999842, -33.6783981, 6.7686806, -40.1239090, 40.2783813
33: -43.7654228, 15.8854160, -43.8616104, 16.0042915, -56.9794159, 56.9627457
34: -50.6146545, -4.3631382, -50.6652374, -4.2459698, -42.2319260, 42.0922546
35: -40.8264084, 6.9498138, -40.8985367, 7.0605950, -43.7686615, 43.7760315
36: -44.4204445, 5.3353062, -44.4343071, 5.4643245, -45.5290756, 45.4714584
37: -59.3481903, 2.3090858, -59.3762817, 2.4062500, -55.1102676, 55.3706970
38: -50.8242188, 8.5233383, -50.8774185, 8.6585655, -59.4827843, 59.4007568
39: -52.0629120, 14.8103170, -52.1000862, 14.9188528, -66.9817657, 66.9104004
40: -47.7567520, 8.2754068, -47.8313599, 8.4001293, -53.0396576, 53.0637665
41: -31.7619514, 15.1651716, -31.8415222, 15.2868195, -45.0807953, 45.3808899
42: -27.0567017, 9.9972363, -27.1621895, 10.1443462, -36.6067734, 36.5599747

Time for backsubstitution: 1.83 seconds

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
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1738
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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1584
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7399635, upper bound: 14.7541368
time: 19.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7399635, upper bound: 14.7752221
time: 41.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.5443344, 32.8749352, -23.6281109, 32.7862244, -53.9016037, 54.0702133
1: -7.4561939, 32.3216667, -7.5213623, 32.2731476, -36.2363892, 36.3387260
2: -4.4874220, 31.7873325, -4.5554304, 31.7041397, -32.8229828, 32.9760361
3: -8.7299337, 28.9663677, -8.7930241, 28.8452053, -32.1984024, 32.3819580
4: -9.7782612, 35.0407791, -9.8793488, 34.9756165, -42.9943695, 43.1626892
5: -10.8857679, 29.9478951, -10.9742146, 29.8403034, -37.8354416, 38.0310364
6: -38.6186905, 7.2707100, -38.5960159, 7.2748289, -44.1062469, 44.1066666
7: -15.0086374, 30.6417198, -15.0956526, 30.5557919, -41.3803101, 41.5466080
8: -15.1003885, 34.5065613, -15.1908121, 34.4481850, -46.6377411, 46.7912216
9: -10.1378555, 27.0332165, -10.1899786, 27.1002007, -35.4809570, 35.4950333
10: -28.2993393, 23.6435795, -28.3274975, 23.7751808, -50.2017822, 50.1116257
11: -35.6895218, 14.0167294, -35.6941071, 14.0991669, -49.1295166, 49.0300217
12: -49.4253693, 1.5298266, -49.3222122, 1.7180657, -43.8046112, 43.5220795
13: -28.7205563, 21.2474594, -28.7044392, 21.2289677, -49.3332062, 49.3355408
14: -70.8266068, -6.6252213, -70.7603302, -6.5586987, -64.2679062, 64.1351089
15: -17.2111168, 24.6001110, -17.2678089, 24.6254120, -41.8365288, 41.8679199
16: -27.2416153, 23.4726467, -27.3548164, 23.5747604, -48.1092834, 48.1315002
17: -71.1588135, -4.0483131, -71.0700989, -4.0048904, -67.1539230, 67.0217896
18: -34.7888298, 11.6096497, -34.6856117, 11.5941296, -40.5354156, 40.4267998
19: -25.6668167, 5.1813374, -25.6300735, 5.1882534, -29.8070564, 29.7637558
20: -26.4398823, 4.2432261, -26.4141273, 4.2606158, -28.9759560, 28.9348183
21: -31.3147278, 9.9739466, -31.2734642, 9.9982405, -40.1980743, 40.1427383
22: -33.5725555, 6.8273678, -33.5290909, 6.8314543, -38.2923508, 38.2425003
23: -26.8783302, 8.7765560, -26.8448105, 8.7897263, -35.1317596, 35.0765305
24: -23.1929054, 9.8106098, -23.1691990, 9.7860584, -32.6559944, 32.6452332
25: -29.1632233, 5.9367566, -29.1397209, 5.9589314, -34.1889496, 34.1356201
26: -43.0423622, 7.6142802, -42.8528061, 7.6593699, -43.6629181, 43.4345627
27: -26.5755653, 11.4055243, -26.5312672, 11.3972836, -37.5906143, 37.5546494
28: -29.5737839, 7.0483766, -29.5204487, 7.0487499, -36.4611359, 36.4160690
29: -32.5774994, 8.8380260, -32.5225067, 8.8476553, -41.4251556, 41.3605347
30: -37.5783920, 6.8072748, -37.5090981, 6.8054190, -44.3838120, 44.3163719
31: -31.3593216, 7.2058811, -31.3314247, 7.2077708, -37.4805031, 37.4472504
32: -33.5960464, 6.5370493, -33.5803871, 6.5906372, -40.1866837, 40.1174355
33: -43.7394218, 15.9068699, -43.8238220, 15.9204798, -56.8539352, 56.9450531
34: -50.5784454, -4.3939424, -50.5987434, -4.3857465, -42.0303802, 42.0645065
35: -40.7952576, 6.9389691, -40.8543472, 6.9644461, -43.6202240, 43.6753769
36: -44.3536949, 5.2464619, -44.3384933, 5.2840009, -45.2772598, 45.2358551
37: -59.3434029, 2.2720666, -59.3134918, 2.2901235, -54.9660950, 54.9589920
38: -50.7738495, 8.4557724, -50.7845726, 8.5049858, -59.2788353, 59.2403450
39: -52.0275841, 14.8359833, -52.0352249, 14.8572140, -66.8847961, 66.8712082
40: -47.7272682, 8.2799911, -47.7457085, 8.2662821, -52.9551239, 52.9921036
41: -31.7403927, 15.1104717, -31.7447796, 15.1183662, -45.2518158, 45.2367859
42: -27.0786152, 9.9311314, -27.0935478, 10.0011778, -36.5234070, 36.4525146

Time for backsubstitution: 1.83 seconds

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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1572
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
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 878
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1675
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
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 772
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
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6972172, upper bound: 14.7520318
time: 36.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6972172, upper bound: 14.7752221
time: 37.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.7581787, 32.9335938, -23.7211895, 32.7895737, -54.1138687, 54.2245178
1: -7.5956039, 32.3638306, -7.5824966, 32.2756538, -36.3685760, 36.4469681
2: -4.6202784, 31.8249969, -4.6139393, 31.7074585, -32.9453964, 33.0771408
3: -8.8299541, 29.0003510, -8.8364811, 28.8495998, -32.2802582, 32.4674988
4: -9.9561920, 35.0895424, -9.9567032, 34.9789963, -43.1626053, 43.2905884
5: -11.0009317, 29.9935493, -11.0244608, 29.8459396, -37.9387512, 38.1353531
6: -38.6625977, 7.3711739, -38.6026688, 7.3178139, -44.2002106, 44.2032013
7: -15.1951876, 30.6900024, -15.1772861, 30.5581036, -41.5361786, 41.6887970
8: -15.2792168, 34.5620537, -15.2681465, 34.4537811, -46.8108444, 46.9317551
9: -10.2777538, 27.0781860, -10.2508669, 27.1042099, -35.6044769, 35.6052780
10: -28.4045982, 23.7002907, -28.3729324, 23.7845840, -50.3078918, 50.2146072
11: -35.7237091, 14.0571270, -35.7078552, 14.1097755, -49.1745911, 49.0891953
12: -49.4777336, 1.7036734, -49.3270874, 1.7918029, -43.9461823, 43.6472015
13: -28.7585716, 21.2803249, -28.7200775, 21.2412338, -49.3838577, 49.4069824
14: -70.9661865, -6.5707321, -70.8221359, -6.5507393, -64.4154510, 64.2514038
15: -17.2856560, 24.6321678, -17.2984924, 24.6304970, -41.9161530, 41.9306602
16: -27.3992119, 23.5235882, -27.4233894, 23.5800743, -48.2542877, 48.2547150
17: -71.2083664, -4.0141296, -71.0907669, -3.9982796, -67.2100830, 67.0766373
18: -34.8138351, 11.6402254, -34.6935158, 11.6062317, -40.5624695, 40.4782181
19: -25.7009525, 5.1953135, -25.6404686, 5.1940460, -29.8572617, 29.7888184
20: -26.4654675, 4.2595825, -26.4209652, 4.2682900, -29.0414925, 28.9516869
21: -31.3591652, 9.9927177, -31.2884369, 10.0056458, -40.2620697, 40.1695480
22: -33.6213722, 6.8871589, -33.5408630, 6.8573217, -38.3690491, 38.3023529
23: -26.9076481, 8.8129616, -26.8525791, 8.8048306, -35.1881256, 35.1234894
24: -23.2180786, 9.8285275, -23.1783161, 9.7951508, -32.6943092, 32.6727409
25: -29.2053623, 6.0144286, -29.1488113, 5.9930024, -34.2673264, 34.2196274
26: -43.0872383, 7.6619191, -42.8622208, 7.6801353, -43.7295151, 43.4776382
27: -26.6067009, 11.4423828, -26.5418472, 11.4082890, -37.6374130, 37.6071320
28: -29.6110802, 7.1170306, -29.5279694, 7.0786371, -36.5302429, 36.4928589
29: -32.6182060, 8.8980436, -32.5331268, 8.8737974, -41.4920044, 41.4311714
30: -37.6012383, 6.8440609, -37.5178146, 6.8191223, -44.4203606, 44.3618774
31: -31.4047794, 7.2365007, -31.3429909, 7.2201934, -37.5686836, 37.4897041
32: -33.6389046, 6.6278820, -33.5876312, 6.6308260, -40.2697296, 40.2155151
33: -43.8163605, 15.9970970, -43.8414955, 15.9603043, -56.9722290, 57.0375137
34: -50.6293488, -4.2822361, -50.6054001, -4.3368134, -42.1442566, 42.1437073
35: -40.8544121, 7.0496664, -40.8638535, 7.0131464, -43.7409668, 43.7552643
36: -44.4093361, 5.3723068, -44.3457947, 5.3394160, -45.3969498, 45.3427353
37: -59.4115219, 2.3387771, -59.3289871, 2.3190837, -55.0741043, 54.9892883
38: -50.8486023, 8.5609064, -50.7957382, 8.5502195, -59.3988228, 59.3566437
39: -52.1055984, 14.8597240, -52.0560837, 14.8668709, -66.9724731, 66.9158096
40: -47.7868881, 8.3130322, -47.7597237, 8.2800226, -53.0266495, 53.0343399
41: -31.7893543, 15.1853256, -31.7553596, 15.1503658, -45.3324127, 45.3142700
42: -27.1148586, 10.0197010, -27.0993862, 10.0390368, -36.6160889, 36.5368729

Time for backsubstitution: 1.83 seconds

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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1431
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7420616, upper bound: 14.7520318
time: 32.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7420616, upper bound: 14.7752221
time: 40.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.5465031, 32.8821793, -23.7016582, 32.8163376, -53.9349976, 54.1508713
1: -7.4571371, 32.3297195, -7.5725775, 32.3007545, -36.2614059, 36.3996124
2: -4.4892902, 31.8032188, -4.6480780, 31.7529221, -32.8711395, 33.0845032
3: -8.7318211, 28.9687157, -8.8410053, 28.8615589, -32.2142181, 32.4353027
4: -9.7798862, 35.0561447, -9.9714317, 35.0256310, -43.0449219, 43.2689972
5: -10.8876419, 29.9521027, -11.0203161, 29.8627396, -37.8577118, 38.0904388
6: -38.6462860, 7.2721071, -38.6839027, 7.4089127, -44.2674484, 44.1910095
7: -15.0101643, 30.6529102, -15.1559505, 30.5907269, -41.4156952, 41.6185684
8: -15.1023560, 34.5218048, -15.2728491, 34.4981384, -46.6864624, 46.8888168
9: -10.1352949, 27.0351334, -10.2188559, 27.1342278, -35.5149536, 35.5305634
10: -28.3014698, 23.6550102, -28.4066525, 23.8270798, -50.2553101, 50.2429581
11: -35.6908875, 14.0262709, -35.7417068, 14.1389179, -49.1716385, 49.0860291
12: -49.4353638, 1.5316877, -49.3606529, 1.7619529, -43.8652115, 43.5660934
13: -28.7235394, 21.2478752, -28.7458324, 21.2564259, -49.3608093, 49.3873062
14: -70.8287201, -6.5998707, -70.9014969, -6.4832191, -64.3455048, 64.3016281
15: -17.2125282, 24.6022358, -17.3015327, 24.6420593, -41.8545876, 41.9037704
16: -27.2427578, 23.4757862, -27.4124069, 23.6012344, -48.1380081, 48.1933289
17: -71.1636658, -4.0081558, -71.2610931, -3.8917351, -67.2719269, 67.2529373
18: -34.7904205, 11.6081038, -34.7285805, 11.6178551, -40.5659180, 40.4657822
19: -25.6683846, 5.1844625, -25.6635208, 5.2113333, -29.8302879, 29.7945480
20: -26.4400463, 4.2444453, -26.4245644, 4.2837248, -28.9977608, 28.9498863
21: -31.3155346, 9.9766483, -31.3038139, 10.0216045, -40.2219849, 40.1705322
22: -33.5744781, 6.8301716, -33.5641479, 6.8566184, -38.3182907, 38.2999039
23: -26.8795815, 8.7788830, -26.8713207, 8.8140125, -35.1581650, 35.1045036
24: -23.1940403, 9.8163090, -23.2259979, 9.8222818, -32.6921387, 32.7058792
25: -29.1635666, 5.9415317, -29.1886444, 5.9937673, -34.2259140, 34.1955643
26: -43.0440941, 7.6135635, -42.8785934, 7.6725030, -43.6836243, 43.4657440
27: -26.5843353, 11.4064026, -26.5774460, 11.4612417, -37.6837769, 37.6050339
28: -29.5745068, 7.0505013, -29.5389519, 7.0911927, -36.5096207, 36.4368668
29: -32.5784798, 8.8409538, -32.5536041, 8.8724194, -41.4508972, 41.3945580
30: -37.5782928, 6.8099680, -37.5236053, 6.8340302, -44.4123230, 44.3335724
31: -31.3616638, 7.2158203, -31.3701324, 7.2482419, -37.5172768, 37.4868774
32: -33.6233025, 6.5378361, -33.6707458, 6.7275238, -40.3508263, 40.2085800
33: -43.7420731, 15.9082336, -43.8479347, 15.9642172, -56.9142303, 56.9779663
34: -50.5954361, -4.3931651, -50.6606522, -4.2950807, -42.1464996, 42.1271133
35: -40.8038673, 6.9396958, -40.8917236, 7.0121107, -43.6838531, 43.7113342
36: -44.3814774, 5.2472000, -44.4263840, 5.4069519, -45.4281311, 45.3237076
37: -59.3543739, 2.2737684, -59.3603477, 2.3732581, -55.0707703, 55.0084076
38: -50.7946320, 8.4571857, -50.8656311, 8.6099167, -59.4045486, 59.3228149
39: -52.0353203, 14.8368912, -52.0795288, 14.9052610, -66.9405823, 66.9164200
40: -47.7492294, 8.2812595, -47.8178596, 8.3846989, -53.0955582, 53.0637283
41: -31.7654877, 15.1115780, -31.8310013, 15.2516394, -45.4113159, 45.3221436
42: -27.0962715, 9.9325285, -27.1561108, 10.1014881, -36.6412430, 36.5132523

Time for backsubstitution: 1.77 seconds

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
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1738
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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 811
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
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1431
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7262839, upper bound: 14.7520318
time: 37.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7262839, upper bound: 14.7752221
time: 37.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.7603683, 32.9408569, -23.7947502, 32.8196564, -54.1472397, 54.3052216
1: -7.5965729, 32.3718910, -7.6337299, 32.3033066, -36.3935699, 36.5077896
2: -4.6221585, 31.8408737, -4.7066031, 31.7562122, -32.9935226, 33.1856308
3: -8.8318319, 29.0027237, -8.8844814, 28.8659782, -32.2960815, 32.5208893
4: -9.9578333, 35.1048965, -10.0487795, 35.0289688, -43.2131271, 43.3968506
5: -11.0027990, 29.9977551, -11.0705738, 29.8683777, -37.9610901, 38.1947479
6: -38.6902084, 7.3725567, -38.6905403, 7.4519348, -44.3613739, 44.2874985
7: -15.1967363, 30.7011948, -15.2376242, 30.5930138, -41.5715714, 41.7607346
8: -15.2811699, 34.5772781, -15.3502064, 34.5037155, -46.8595276, 47.0293579
9: -10.2751808, 27.0801010, -10.2797585, 27.1382637, -35.6384964, 35.6408348
10: -28.4066906, 23.7117538, -28.4521523, 23.8363934, -50.3613586, 50.3459320
11: -35.7251625, 14.0666418, -35.7554626, 14.1495438, -49.2166214, 49.1452942
12: -49.4876442, 1.7055044, -49.3655319, 1.8357558, -44.0068665, 43.6912231
13: -28.7615757, 21.2807484, -28.7615051, 21.2687645, -49.4115448, 49.4587708
14: -70.9682693, -6.5453491, -70.9632950, -6.4753189, -64.4929504, 64.4179459
15: -17.2870674, 24.6343040, -17.3322086, 24.6471519, -41.9342194, 41.9665146
16: -27.4003906, 23.5266647, -27.4810104, 23.6065712, -48.2830658, 48.3165741
17: -71.2132721, -3.9740334, -71.2817078, -3.8852158, -67.3280563, 67.3076782
18: -34.8154564, 11.6386814, -34.7364540, 11.6299210, -40.5929756, 40.5171738
19: -25.7025261, 5.1984539, -25.6739464, 5.2171288, -29.8804855, 29.8196182
20: -26.4656277, 4.2608171, -26.4314117, 4.2914047, -29.0633087, 28.9667816
21: -31.3599892, 9.9955158, -31.3187752, 10.0289993, -40.2859650, 40.1973572
22: -33.6233177, 6.8899555, -33.5759163, 6.8824573, -38.3949738, 38.3598328
23: -26.9088955, 8.8152609, -26.8790932, 8.8291054, -35.2145309, 35.1514626
24: -23.2192116, 9.8342094, -23.2351151, 9.8313484, -32.7304077, 32.7333641
25: -29.2057381, 6.0192261, -29.1977272, 6.0278277, -34.3042831, 34.2795525
26: -43.0888672, 7.6612163, -42.8880043, 7.6932745, -43.7501984, 43.5087967
27: -26.6154823, 11.4432659, -26.5879898, 11.4722614, -37.7305527, 37.6575508
28: -29.6117897, 7.1191335, -29.5465126, 7.1211252, -36.5787582, 36.5137177
29: -32.6192551, 8.9009609, -32.5642471, 8.8985529, -41.5178070, 41.4652100
30: -37.6011581, 6.8468046, -37.5323257, 6.8476667, -44.4488258, 44.3791313
31: -31.4071465, 7.2464180, -31.3816662, 7.2606354, -37.6054420, 37.5293045
32: -33.6661415, 6.6286297, -33.6780205, 6.7677422, -40.4338837, 40.3066483
33: -43.8189774, 15.9984322, -43.8655701, 16.0040588, -57.0326691, 57.0703583
34: -50.6462898, -4.2814507, -50.6672897, -4.2461491, -42.2604294, 42.2063599
35: -40.8630562, 7.0504451, -40.9012642, 7.0608039, -43.8046112, 43.7911835
36: -44.4370766, 5.3730412, -44.4336472, 5.4623470, -45.5478363, 45.4304810
37: -59.4224892, 2.3404250, -59.3758507, 2.4021959, -55.1787567, 55.0387955
38: -50.8693619, 8.5622396, -50.8767662, 8.6551733, -59.5245361, 59.4390068
39: -52.1133270, 14.8606653, -52.1003723, 14.9150467, -67.0283737, 66.9610367
40: -47.8088837, 8.3143606, -47.8318825, 8.3984098, -53.1671448, 53.1059418
41: -31.8144913, 15.1863928, -31.8415432, 15.2836323, -45.4919128, 45.3995667
42: -27.1325169, 10.0210991, -27.1619434, 10.1393423, -36.7339172, 36.5975571

Time for backsubstitution: 1.84 seconds

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
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1738
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
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1448
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
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1461
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
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1431
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7711206, upper bound: 14.7520318
time: 41.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.7711206, upper bound: 14.7752221
time: 47.06 seconds

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

Time for backsubstitution: 1.84 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
time: 37.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 33.61 seconds

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

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
time: 30.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
time: 31.93 seconds

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

Time for backsubstitution: 1.83 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6951250, upper bound: 14.7919458
time: 36.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -14.6951250, upper bound: 14.8130237
time: 44.23 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 82.69 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7541369
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7541369
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7541368
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.7399635, upper bound: 14.7541368
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.7399635, upper bound: 14.7752221
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6972172, upper bound: 14.7520318
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6972172, upper bound: 14.7752221
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.7420616, upper bound: 14.7520318
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.7420616, upper bound: 14.7752221
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.7262839, upper bound: 14.7520318
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.7262839, upper bound: 14.7752221
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.7711206, upper bound: 14.7520318
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.7711206, upper bound: 14.7752221
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7919458
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6951250, upper bound: 14.7919458
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 82.69
Output dim: 2, lower bound: -14.6951250, upper bound: 14.8130237
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6972172, upper bound: 14.8130237
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7420616, upper bound: 14.8130237
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7262839, upper bound: 14.8130237
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7711206, upper bound: 14.8130237
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7204540, upper bound: 14.7752221
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.7752221
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7391718, upper bound: 14.7752221
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7840025, upper bound: 14.7752221
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7682095, upper bound: 14.7752221
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.8130231, upper bound: 14.7752221
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7204540, upper bound: 14.8130237
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7391718, upper bound: 14.8130237
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6660539, upper bound: 14.8130237
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.7682095, upper bound: 14.8130237
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 82.69
Output dim: 2, lower bound: -14.6913915, upper bound: 14.8130237

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 50.94 + 3566.96 = 3617.90 seconds
