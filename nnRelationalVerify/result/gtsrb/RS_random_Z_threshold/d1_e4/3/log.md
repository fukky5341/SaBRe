## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 1800 seconds
Split limit: 100
Threshold: 10.5070886937


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2095375, 18.2095375)
1: (-13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9025345, 11.9025345)
2: (-12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6323814, 10.6323814)
3: (-21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2811050, 16.2811089)
4: (-19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5144386, 14.5144424)
5: (-15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4280739, 15.4280739)
6: (-21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7600861, 16.7600899)
7: (-18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0078735, 17.0078735)
8: (-28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9188309, 17.9188271)
9: (-19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9140854, 17.9140854)
10: (-16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8991394, 19.8991432)
11: (-2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1921921, 17.1921921)
12: (-17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3953476, 24.3953476)
13: (-30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9720306, 20.9720345)
14: (-34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3704453, 29.3704453)
15: (-15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6508636, 18.6508636)
16: (-15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2957535, 19.2957573)
17: (-23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2987366, 23.2987366)
18: (1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8489532, 18.8489532)
19: (-0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0614567, 11.0614548)
20: (-4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3720016, 13.3720016)
21: (-1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0370331, 16.0370331)
22: (-3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5892448, 13.5892448)
23: (-1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2790985, 13.2790966)
24: (-1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0926666, 15.0926628)
25: (-2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5594864, 17.5594864)
26: (-5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4541855, 25.4541855)
27: (-0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5211563, 13.5211601)
28: (-1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1801147, 14.1801147)
29: (-2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2860107, 11.2860146)
30: (-8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1624680, 20.1624680)
31: (0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3200417, 14.3200417)
32: (-22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9594345, 18.9594345)
33: (-39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0123825, 21.0123863)
34: (-33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5314941, 17.5314980)
35: (-24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7993317, 18.7993355)
36: (-20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1306381, 20.1306458)
37: (-32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0276031, 26.0275955)
38: (-28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4695053, 24.4695053)
39: (-44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1526794, 23.1526794)
40: (-31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9793434, 12.9793434)
41: (-19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8033371, 18.8033371)
42: (-20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6033325, 13.6033325)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.07 + 32.04 = 35.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 18, lower bound: -10.5176063, upper bound: 10.5176063

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1006

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1664

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5172343, upper bound: 10.5172829
time: 20.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5172828, upper bound: 10.5172343
time: 22.18 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 43.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 43.18
Output dim: 18, lower bound: -10.5172343, upper bound: 10.5172829
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 43.18
Output dim: 18, lower bound: -10.5172828, upper bound: 10.5172343

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2071571, 18.2074890
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9017982, 11.9018822
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6323013, 10.6323414
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2808380, 16.2808228
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5136414, 14.5136528
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4279823, 15.4280128
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7582397, 16.7574158
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0077591, 17.0078888
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9187012, 17.9187126
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9139938, 17.9139862
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8989716, 19.8988647
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1914444, 17.1921043
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3938370, 24.3931274
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9718628, 20.9718742
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3707428, 29.3704529
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6508179, 18.6508255
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2956963, 19.2956314
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2995300, 23.2987366
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8480225, 18.8485184
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0611706, 11.0612602
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3719749, 13.3719826
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0368118, 16.0369072
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5893898, 13.5891991
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2763405, 13.2772160
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0911026, 15.0914459
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5590134, 17.5590439
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4534912, 25.4539337
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5208054, 13.5210876
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1801186, 14.1801109
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2860107, 11.2859955
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1624146, 20.1623688
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3200378, 14.3200340
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9580078, 18.9577103
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0114098, 21.0104370
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5301514, 17.5293083
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7984238, 18.7977753
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1306076, 20.1306534
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0271454, 26.0275574
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4694977, 24.4704514
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1516113, 23.1516571
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9781265, 12.9777374
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8033371, 18.8033752
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6032639, 13.6032257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 982

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1481

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5154866, upper bound: 10.5163646
time: 21.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5163160, upper bound: 10.5155351
time: 21.92 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2074928, 18.2071609
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9018822, 11.9017982
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6323395, 10.6323032
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2808228, 16.2808380
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5136566, 14.5136414
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4280128, 15.4279823
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7574158, 16.7582397
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0078888, 17.0077591
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9187088, 17.9186974
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9139862, 17.9139938
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8988647, 19.8989716
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1921005, 17.1914482
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3931274, 24.3938370
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9718704, 20.9718628
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3704529, 29.3707504
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6508179, 18.6508217
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2956352, 19.2957001
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2987366, 23.2995300
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8485184, 18.8480225
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0612583, 11.0611725
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3719826, 13.3719730
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0369034, 16.0368156
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5891991, 13.5893936
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2772141, 13.2763405
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0914459, 15.0911064
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5590439, 17.5590134
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4539337, 25.4534912
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5210876, 13.5208015
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1801109, 14.1801186
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2859955, 11.2860107
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1623688, 20.1624146
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3200340, 14.3200378
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9577103, 18.9580078
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0104408, 21.0114059
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5293045, 17.5301476
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7977753, 18.7984238
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1306534, 20.1306076
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0275574, 26.0271378
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4704514, 24.4694977
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1516571, 23.1516151
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9777374, 12.9781265
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8033752, 18.8033371
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6032257, 13.6032639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1023

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1351

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5166667, upper bound: 10.5052835
time: 25.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5052835, upper bound: 10.5166178
time: 21.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 49.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 49.68
Output dim: 18, lower bound: -10.5154866, upper bound: 10.5163646
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 49.68
Output dim: 18, lower bound: -10.5163160, upper bound: 10.5155351
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 49.68
Output dim: 18, lower bound: -10.5166667, upper bound: 10.5052835
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 49.68
Output dim: 18, lower bound: -10.5052835, upper bound: 10.5166178

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2073364, 18.2070923
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9018021, 11.9016838
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6323738, 10.6320400
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2808075, 16.2807274
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5136147, 14.5135117
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4279480, 15.4278984
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7579651, 16.7574043
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0077286, 17.0076752
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9187927, 17.9183731
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9137955, 17.9137573
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8989258, 19.8987808
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1908569, 17.1918144
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3933640, 24.3931351
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9715576, 20.9713402
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3706665, 29.3700409
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6508636, 18.6507797
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2955971, 19.2956886
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2995300, 23.2985077
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8477325, 18.8483925
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0611153, 11.0611610
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3717842, 13.3718643
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0367584, 16.0370598
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5893593, 13.5891647
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2760277, 13.2770729
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0911713, 15.0912971
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5589676, 17.5590057
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4533310, 25.4538269
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5202332, 13.5206985
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1798325, 14.1799850
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2859688, 11.2859688
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1623840, 20.1623535
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3200226, 14.3199997
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9577789, 18.9578323
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0106163, 21.0104370
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5297012, 17.5289879
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7978134, 18.7971878
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1303940, 20.1303635
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0265961, 26.0274277
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4695053, 24.4703522
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1509781, 23.1504745
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9777222, 12.9776764
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8031158, 18.8033371
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6030693, 13.6031799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 907

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5130424, upper bound: 10.5139337
time: 12.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5130612, upper bound: 10.5139148
time: 21.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2067566, 18.2074890
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9015999, 11.9018822
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6320038, 10.6323414
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2808380, 16.2807961
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5135002, 14.5136528
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4278717, 15.4280128
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7582397, 16.7571373
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0075455, 17.0078888
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9183578, 17.9187126
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9139938, 17.9137917
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8989716, 19.8988228
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1911469, 17.1921043
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3938370, 24.3926544
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9718628, 20.9715691
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3703308, 29.3704529
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6507721, 18.6508255
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2956963, 19.2955246
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2993011, 23.2987366
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8479004, 18.8485184
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0610733, 11.0612602
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3718567, 13.3719826
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0368118, 16.0368462
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5893898, 13.5891685
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2761955, 13.2772160
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0909576, 15.0914459
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5589752, 17.5590439
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4533920, 25.4539337
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5204163, 13.5210876
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1799927, 14.1801109
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2859879, 11.2859955
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1624069, 20.1623688
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3200035, 14.3200340
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9580078, 18.9574814
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0114098, 21.0096512
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5301514, 17.5288620
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7984238, 18.7971611
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1306076, 20.1304474
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0271454, 26.0270233
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4693909, 24.4704514
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1516113, 23.1510162
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9781265, 12.9773331
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8033371, 18.8031540
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6032639, 13.6030312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1583

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5109732, upper bound: 10.5091048
time: 19.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5098845, upper bound: 10.5101927
time: 19.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2057571, 18.2053947
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9002457, 11.9000969
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6351662, 10.6347961
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2775955, 16.2779617
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5140991, 14.5141144
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4283142, 15.4282951
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7538681, 16.7551231
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0095062, 17.0092468
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9188919, 17.9185371
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9119186, 17.9121475
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8969383, 19.8971138
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1925278, 17.1914825
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3860321, 24.3873520
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9627914, 20.9636154
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3686523, 29.3687134
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6509094, 18.6509056
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2956581, 19.2957344
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2982407, 23.2989731
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8437729, 18.8426781
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0581818, 11.0577831
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3711166, 13.3708305
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0359077, 16.0357971
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5896225, 13.5898247
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2745209, 13.2733135
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0876465, 15.0868340
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5586014, 17.5584946
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4492569, 25.4482346
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5156708, 13.5147438
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1774826, 14.1771660
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2844505, 11.2842789
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1619263, 20.1619186
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3172569, 14.3170166
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9519119, 18.9528847
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0174637, 21.0197678
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5373383, 17.5394058
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.8048401, 18.8065109
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1227112, 20.1232071
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0320892, 26.0323334
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4657135, 24.4652252
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1374893, 23.1386185
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9729919, 12.9739189
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8002014, 18.8004990
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6000099, 13.6004372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1640

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1768

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4976560, upper bound: 10.5041158
time: 23.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5155071, upper bound: 10.4862642
time: 21.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2057266, 18.2054253
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9001808, 11.9001656
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6348343, 10.6351280
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2779465, 16.2776146
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5141296, 14.5140915
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4283218, 15.4282875
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7543030, 16.7546921
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0093765, 17.0093765
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9185486, 17.9188766
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9121399, 17.9119263
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8970146, 19.8970413
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1921463, 17.1918716
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3866425, 24.3867416
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9636307, 20.9627876
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3684082, 29.3689575
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6509094, 18.6509094
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2956657, 19.2957268
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2981796, 23.2990341
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8431778, 18.8432693
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0578690, 11.0580940
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3708420, 13.3711052
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0358849, 16.0358162
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5896301, 13.5898132
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2741890, 13.2736454
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0871735, 15.0873032
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5585251, 17.5585709
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4486771, 25.4488144
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5150299, 13.5153923
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1771545, 14.1774902
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2842598, 11.2844696
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1618652, 20.1619797
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3170166, 14.3172607
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9525909, 18.9522095
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0187988, 21.0184326
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5385590, 17.5381851
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.8058624, 18.8054924
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1232529, 20.1226578
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0327606, 26.0316620
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4661865, 24.4647598
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1386642, 23.1374512
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9735260, 12.9733849
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8005371, 18.8001633
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6003990, 13.6000481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1681

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1732

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4966499, upper bound: 10.5049073
time: 25.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4936390, upper bound: 10.5079176
time: 29.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 56.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 56.52
Output dim: 18, lower bound: -10.5130424, upper bound: 10.5139337
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 56.52
Output dim: 18, lower bound: -10.5130612, upper bound: 10.5139148
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 56.52
Output dim: 18, lower bound: -10.5109732, upper bound: 10.5091048
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 56.52
Output dim: 18, lower bound: -10.5098845, upper bound: 10.5101927
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 56.52
Output dim: 18, lower bound: -10.4976560, upper bound: 10.5041158
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 56.52
Output dim: 18, lower bound: -10.5155071, upper bound: 10.4862642
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 56.52
Output dim: 18, lower bound: -10.4966499, upper bound: 10.5049073
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 56.52
Output dim: 18, lower bound: -10.4936390, upper bound: 10.5079176

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2062836, 18.2055969
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9012794, 11.9011269
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6321716, 10.6320496
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2804756, 16.2801819
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5133858, 14.5136375
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4278259, 15.4277725
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7566833, 16.7554626
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0076981, 17.0076370
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9187012, 17.9182625
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9137344, 17.9136810
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8988495, 19.8986969
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1907578, 17.1917763
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3931122, 24.3928070
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9715271, 20.9712677
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3688812, 29.3686905
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6508560, 18.6507607
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2947578, 19.2948723
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2984009, 23.2979965
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8475189, 18.8483887
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0592041, 11.0597973
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3717613, 13.3718414
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0354996, 16.0360985
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5889587, 13.5890007
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2756538, 13.2767200
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0911484, 15.0913239
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5589294, 17.5589828
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4523697, 25.4531860
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5200768, 13.5205574
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1793175, 14.1796341
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2857246, 11.2857475
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1623611, 20.1623383
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3181953, 14.3186874
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9564209, 18.9561386
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0090942, 21.0086708
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5273209, 17.5259933
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7978134, 18.7971764
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1299973, 20.1295319
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0248871, 26.0252991
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4676666, 24.4673004
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1508255, 23.1503067
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9764290, 12.9759636
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8010788, 18.8008270
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6008606, 13.5995979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1427

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5113924, upper bound: 10.5117074
time: 23.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5108161, upper bound: 10.5122836
time: 23.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2058411, 18.2060318
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9012451, 11.9011574
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6323853, 10.6318359
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2802620, 16.2803955
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5137444, 14.5132751
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4278259, 15.4277763
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7560196, 16.7561264
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0076904, 17.0076447
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9186859, 17.9182777
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9137268, 17.9136925
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8988419, 19.8987045
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1908188, 17.1917191
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3930435, 24.3928833
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9714813, 20.9713097
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3693085, 29.3682404
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6508484, 18.6507683
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2947731, 19.2948532
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2990265, 23.2973785
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8477325, 18.8481827
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0597496, 11.0592518
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3717613, 13.3718414
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0357895, 16.0357971
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5891953, 13.5887642
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2756729, 13.2767010
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0911942, 15.0912781
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5589447, 17.5589752
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4526978, 25.4528580
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5200996, 13.5205383
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1794815, 14.1794662
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2857475, 11.2857246
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1623688, 20.1623306
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3187103, 14.3181763
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9560852, 18.9564743
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0088501, 21.0089149
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5267029, 17.5266075
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7977982, 18.7971878
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1295700, 20.1299591
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0244751, 26.0257111
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4664612, 24.4685059
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1508026, 23.1503258
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9760094, 12.9763832
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8005981, 18.8013077
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5994911, 13.6009712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 991

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 982

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5128518, upper bound: 10.5137127
time: 26.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5128518, upper bound: 10.5137127
time: 26.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2049713, 18.2045746
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9005432, 11.9004059
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6315765, 10.6308956
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2807007, 16.2804451
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5134315, 14.5132675
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4273911, 15.4266510
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7557220, 16.7561111
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0073090, 17.0074310
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9174042, 17.9147873
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9139023, 17.9139938
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8980827, 19.8953018
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1902657, 17.1900749
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3904114, 24.3919067
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9645309, 20.9697647
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3695374, 29.3695831
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6502380, 18.6489754
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2954445, 19.2949295
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2989807, 23.2981567
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8487167, 18.8478088
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0610580, 11.0610218
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3718452, 13.3719120
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0367737, 16.0370026
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5889473, 13.5893745
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2748528, 13.2737198
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0910339, 15.0904350
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5584869, 17.5588417
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4541473, 25.4528961
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5201149, 13.5201302
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1802101, 14.1794052
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2856674, 11.2847672
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1623383, 20.1621323
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3190346, 14.3196716
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9535904, 18.9562149
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0042534, 21.0076294
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5285797, 17.5278130
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7941971, 18.7960587
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1255341, 20.1292343
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0262146, 26.0268250
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4662857, 24.4697418
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1417007, 23.1483498
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9765015, 12.9765472
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8032990, 18.8031540
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6015587, 13.6020241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1581

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5069447, upper bound: 10.5086376
time: 19.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5105519, upper bound: 10.5049890
time: 23.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2038498, 18.2056961
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9001198, 11.9008293
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6305580, 10.6319141
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2804871, 16.2806549
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5131111, 14.5135803
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4265289, 15.4275131
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7572098, 16.7546196
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0070801, 17.0076599
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9144363, 17.9177589
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9141998, 17.9136925
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8954582, 19.8979301
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1891212, 17.1912231
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3930740, 24.3892441
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9700470, 20.9642487
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3694763, 29.3696594
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6489258, 18.6502838
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2951012, 19.2952728
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2987213, 23.2984161
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8471756, 18.8493423
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0608368, 11.0612411
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3717842, 13.3719711
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0369720, 16.0368042
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5895805, 13.5887375
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2727089, 13.2758636
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0899353, 15.0915413
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5587769, 17.5585556
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4523621, 25.4546814
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5194588, 13.5207863
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1792831, 14.1803284
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2847633, 11.2856712
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1621628, 20.1623077
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3196335, 14.3190689
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9567337, 18.9530678
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0093956, 21.0024872
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5290985, 17.5273018
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7973022, 18.7929535
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1294022, 20.1253662
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0269318, 26.0260925
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4686813, 24.4673462
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1489563, 23.1410942
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9773407, 12.9757080
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8033371, 18.8031235
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6022606, 13.6013222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1676

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 982

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5097017, upper bound: 10.5100062
time: 24.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5097017, upper bound: 10.5100062
time: 25.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1933174, 18.1969261
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8926125, 11.8948593
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6272964, 10.6293983
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2706680, 16.2732658
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4994087, 14.5040283
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4192238, 15.4220543
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7533646, 16.7546997
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9995270, 17.0024033
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9028549, 17.9075317
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9038544, 17.9066086
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8969154, 19.8970795
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1859055, 17.1818428
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3817062, 24.3800354
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9614563, 20.9636917
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3666000, 29.3679657
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6455994, 18.6472588
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2952805, 19.2967339
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2970963, 23.2990952
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8401031, 18.8372917
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0564537, 11.0551796
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3705406, 13.3705902
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0325203, 16.0307198
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5891724, 13.5891037
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2680473, 13.2638836
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0847015, 15.0824394
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5564499, 17.5553589
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4471970, 25.4452515
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5149536, 13.5136032
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1744156, 14.1726952
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2815819, 11.2800026
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1578598, 20.1559906
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3160667, 14.3147316
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9523849, 18.9527550
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0172348, 21.0194855
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5336456, 17.5339317
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.8009186, 18.8007889
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1215439, 20.1217422
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0287628, 26.0274353
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4656448, 24.4653168
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1370010, 23.1396484
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9726257, 12.9740334
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7998810, 18.7990723
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6000099, 13.6002617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1614

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1660

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5097693, upper bound: 10.4851233
time: 23.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5143591, upper bound: 10.4805366
time: 26.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2057266, 18.2042389
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9001808, 11.8997574
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6348343, 10.6339073
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2779465, 16.2769279
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5141296, 14.5126991
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4283218, 15.4279137
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7541733, 16.7546921
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0093765, 17.0085602
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9185486, 17.9172516
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9121399, 17.9115067
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8970146, 19.8969879
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1919098, 17.1918716
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3863754, 24.3867416
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9636307, 20.9619484
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3684082, 29.3687820
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6509094, 18.6508522
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2956657, 19.2950439
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2981796, 23.2989273
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8424988, 18.8432693
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0578194, 11.0580940
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3703041, 13.3711052
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0354538, 16.0358162
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5890350, 13.5898132
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2738113, 13.2736454
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0865097, 15.0873032
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5575867, 17.5585709
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4482956, 25.4488144
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5145264, 13.5153923
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1762123, 14.1774902
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2830048, 11.2844696
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1601105, 20.1619797
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3170128, 14.3172607
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9525909, 18.9518394
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0187988, 21.0180550
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5383301, 17.5381851
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.8058624, 18.8054161
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1232529, 20.1224365
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0327606, 26.0311279
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4661865, 24.4644928
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1386642, 23.1359444
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9735260, 12.9721642
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8005371, 18.7998047
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6003952, 13.6000481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 990

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1023

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4936293, upper bound: 10.5067104
time: 26.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4924290, upper bound: 10.5079080
time: 13.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 42.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5113924, upper bound: 10.5117074
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5108161, upper bound: 10.5122836
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5128518, upper bound: 10.5137127
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5128518, upper bound: 10.5137127
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5069447, upper bound: 10.5086376
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5105519, upper bound: 10.5049890
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5097017, upper bound: 10.5100062
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5097017, upper bound: 10.5100062
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5097693, upper bound: 10.4851233
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.5143591, upper bound: 10.4805366
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.4936293, upper bound: 10.5067104
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.34
Output dim: 18, lower bound: -10.4924290, upper bound: 10.5079080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2044830, 18.2029533
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9028511, 11.9023743
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6311188, 10.6307640
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2845917, 16.2835617
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5150452, 14.5150909
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4294205, 15.4289970
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7470169, 16.7468109
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0091782, 17.0088959
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9201202, 17.9194260
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9176178, 17.9169083
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8990288, 19.8988457
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1773071, 17.1782990
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3889236, 24.3891983
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9570465, 20.9582100
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3700638, 29.3694000
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6478958, 18.6475220
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2955856, 19.2955475
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2992325, 23.2987518
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8419800, 18.8423386
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0528412, 11.0533123
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3687973, 13.3689289
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0297852, 16.0298958
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5889549, 13.5890694
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2729721, 13.2731667
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0828629, 15.0831490
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5579910, 17.5582848
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4597855, 25.4594879
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5125084, 13.5132484
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1784668, 14.1786652
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2891464, 11.2891426
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1596298, 20.1599121
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3104057, 14.3115234
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9459991, 18.9468079
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9842606, 20.9859619
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5083160, 17.5085831
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7866135, 18.7870064
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1209259, 20.1213531
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0147247, 26.0161591
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4586945, 24.4593277
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1277618, 23.1292648
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9643517, 12.9652557
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7955780, 18.7959518
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5917320, 13.5903702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1567

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1693

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5085915, upper bound: 10.5044204
time: 23.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5041071, upper bound: 10.5089053
time: 23.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.2036438, 18.2038002
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9025307, 11.9026985
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6308861, 10.6309967
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2838593, 16.2842979
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5148392, 14.5152969
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4290543, 15.4293633
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7480316, 16.7457962
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0089645, 17.0091171
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9198608, 17.9196854
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9169617, 17.9175644
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8989983, 19.8988762
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1772842, 17.1783257
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3894958, 24.3886185
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9584808, 20.9567833
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3695908, 29.3698883
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6476135, 18.6478043
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2954330, 19.2957077
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2991562, 23.2988281
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8414688, 18.8428421
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0527191, 11.0534325
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3688469, 13.3688812
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0292816, 16.0303917
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5890312, 13.5889931
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2720985, 13.2740364
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0829697, 15.0830383
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5582352, 17.5580444
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4586716, 25.4606094
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5127678, 13.5129929
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1783485, 14.1787834
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2891235, 11.2891693
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1599350, 20.1596069
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3110313, 14.3108978
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9470901, 18.9457169
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9863892, 20.9838333
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5099106, 17.5069847
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7876434, 18.7859802
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1218185, 20.1204605
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0157471, 26.0151367
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4596939, 24.4583282
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1297836, 23.1272392
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9657211, 12.9638863
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7962036, 18.7953262
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5916328, 13.5904655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 958

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4970247, upper bound: 10.5122285
time: 20.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5107610, upper bound: 10.4984943
time: 22.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1863976, 18.1889915
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8953056, 11.8964081
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6242676, 10.6247501
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2675552, 16.2691917
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5050621, 14.5057335
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4151802, 15.4166794
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7441864, 16.7427139
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0015182, 17.0022888
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9045944, 17.9062729
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9120636, 17.9119720
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8988495, 19.9002953
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1906967, 17.1916046
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3824692, 24.3808136
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9713745, 20.9710960
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3540192, 29.3549347
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6395950, 18.6409302
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2929764, 19.2923965
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2909012, 23.2905807
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8466034, 18.8470993
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0591240, 11.0585060
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3723412, 13.3724308
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0372543, 16.0368195
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5883484, 13.5878792
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2749481, 13.2760391
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0899048, 15.0903625
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5586624, 17.5586853
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4524536, 25.4526443
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5218887, 13.5224953
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1800079, 14.1798325
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2832260, 11.2833786
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1629791, 20.1627197
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3211060, 14.3199959
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9464645, 18.9454002
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0007706, 20.9991455
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5273438, 17.5271454
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7969894, 18.7961578
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1261368, 20.1259613
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0127487, 26.0122986
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4725800, 24.4757004
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1410980, 23.1397247
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9594841, 12.9576225
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7927322, 18.7923889
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5950623, 13.5960388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1582

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4932484, upper bound: 10.5028739
time: 27.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5019958, upper bound: 10.4941247
time: 21.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1888008, 18.1865845
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8964958, 11.8952179
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6252975, 10.6237202
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2690659, 16.2676811
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5062065, 14.5045929
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4167213, 15.4151382
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7426071, 16.7442932
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0023346, 17.0014801
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9066772, 17.9041862
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9120026, 17.9120255
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.9004288, 19.8987122
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1907043, 17.1916008
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3809738, 24.3823090
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9712677, 20.9711990
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3560028, 29.3529510
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6410141, 18.6395073
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2923203, 19.2930527
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2922363, 23.2892456
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8466492, 18.8470535
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0590057, 11.0586243
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3723526, 13.3724174
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0368118, 16.0372620
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5883102, 13.5879135
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2750092, 13.2759762
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0902786, 15.0899849
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5586548, 17.5586929
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4524841, 25.4526138
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5220566, 13.5223312
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1798477, 14.1799927
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2834015, 11.2832031
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1627579, 20.1629486
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3205338, 14.3205719
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9450150, 18.9468536
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9990768, 21.0008354
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5272446, 17.5272446
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7967682, 18.7963791
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1255722, 20.1265259
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0110703, 26.0139694
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4736481, 24.4746246
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1401978, 23.1406174
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9572487, 12.9598541
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7916794, 18.7934341
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5945587, 13.5965424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1292

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5115864, upper bound: 10.5124460
time: 27.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5115864, upper bound: 10.5124460
time: 28.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1957703, 18.1935883
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.9004059, 11.8995705
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6291580, 10.6285114
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2810059, 16.2807465
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5068207, 14.5068741
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4202957, 15.4198380
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7534027, 16.7538452
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0079269, 17.0080185
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9056320, 17.9035263
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9100952, 17.9083023
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8871498, 19.8853302
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1899834, 17.1897774
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3787537, 24.3788986
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9322586, 20.9333954
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3711853, 29.3710022
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6439362, 18.6428986
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2923851, 19.2915764
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2995377, 23.3000031
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8459473, 18.8460846
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0631046, 11.0635452
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3689194, 13.3692932
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0393829, 16.0392838
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5827255, 13.5838661
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2746239, 13.2734871
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0881271, 15.0885010
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5585785, 17.5590172
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4476776, 25.4472885
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5138397, 13.5144272
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1824303, 14.1823006
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2755928, 11.2757988
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1503983, 20.1516724
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3149376, 14.3161964
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9443741, 18.9452972
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9811592, 20.9815598
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5284500, 17.5276680
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7807999, 18.7809639
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1025543, 20.1033478
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0060577, 26.0041122
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4348602, 24.4343567
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1053391, 23.1074905
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9712257, 12.9707985
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7943420, 18.7932587
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5930214, 13.5921936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 975

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5023236, upper bound: 10.5086125
time: 24.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5069197, upper bound: 10.5040168
time: 26.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1939697, 18.1953773
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8997116, 11.9002647
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6291885, 10.6284809
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2809906, 16.2807541
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5070343, 14.5066643
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4205780, 15.4195633
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7534561, 16.7537994
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0079117, 17.0080338
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9061432, 17.9030113
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9082031, 17.9101982
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8881111, 19.8843765
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1899681, 17.1897812
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3774185, 24.3802414
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9281616, 20.9374924
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3709412, 29.3712463
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6441650, 18.6426735
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2920952, 19.2918625
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.3008423, 23.2986984
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8470001, 18.8450279
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0635777, 11.0630760
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3692207, 13.3689919
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0390549, 16.0396118
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5834885, 13.5831413
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2746086, 13.2735023
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0891037, 15.0875206
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5586853, 17.5589142
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4485321, 25.4464264
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5144196, 13.5138550
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1831017, 14.1816292
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2766991, 11.2746925
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1518860, 20.1501923
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3155708, 14.3155632
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9426804, 18.9469986
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9781914, 20.9845276
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5284500, 17.5276718
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7791214, 18.7826424
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0996628, 20.1062393
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0035095, 26.0066681
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4308777, 24.4383392
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1008301, 23.1120033
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9707489, 12.9713020
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7934113, 18.7941895
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5917206, 13.5934944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4934527, upper bound: 10.4897464
time: 24.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4951433, upper bound: 10.4880521
time: 19.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1844101, 18.1886711
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8941879, 11.8960800
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6224442, 10.6248417
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2677841, 16.2694511
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5044365, 14.5060425
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4138870, 15.4164238
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7453613, 16.7411995
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0009308, 17.0023041
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9003372, 17.9057503
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9125443, 17.9119797
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8954391, 19.8995018
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1890068, 17.1911087
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3824844, 24.3771744
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9699478, 20.9640388
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3541718, 29.3563614
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6376724, 18.6404495
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2933044, 19.2928123
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2905960, 23.2915955
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8460388, 18.8482475
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0602188, 11.0605087
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3723640, 13.3725624
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0384369, 16.0378265
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5887413, 13.5878601
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2719841, 13.2751999
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0886307, 15.0906143
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5584869, 17.5582542
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4521179, 25.4544678
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5212593, 13.5227470
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1798134, 14.1806984
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2822380, 11.2833252
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1627731, 20.1627045
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3220291, 14.3208885
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9471207, 18.9420013
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0013390, 20.9927254
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5297279, 17.5278320
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7965012, 18.7919235
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1259766, 20.1213684
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0151978, 26.0126953
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4748001, 24.4745331
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1392288, 23.1304855
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9608231, 12.9569511
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7954788, 18.7941971
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5978203, 13.5963860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 760

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5095822, upper bound: 10.4987024
time: 23.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4984002, upper bound: 10.5098867
time: 23.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1868134, 18.1862602
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8953781, 11.8948898
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6234741, 10.6238117
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2692947, 16.2679405
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5055733, 14.5049019
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4154282, 15.4148865
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7437820, 16.7427750
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0017395, 17.0014877
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9024200, 17.9036636
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9124832, 17.9120331
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8970261, 19.8979225
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1890144, 17.1911087
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3809891, 24.3786621
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9698410, 20.9641457
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3561554, 29.3543777
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6391068, 18.6390266
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2926483, 19.2934647
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2919312, 23.2902679
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8460846, 18.8481979
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0601006, 11.0606251
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3723755, 13.3725510
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0379944, 16.0382690
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5887032, 13.5878906
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2720451, 13.2751369
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0890045, 15.0902367
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5584869, 17.5582619
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4521484, 25.4544373
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5214272, 13.5225792
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1796494, 14.1808586
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2824135, 11.2831497
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1625519, 20.1629257
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3214569, 14.3214607
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9456711, 18.9434547
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9996452, 20.9944153
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5296288, 17.5279312
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7962875, 18.7921448
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1254120, 20.1219330
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0135345, 26.0143585
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4758682, 24.4734573
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1383286, 23.1313782
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9585876, 12.9591866
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7944260, 18.7952499
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5973129, 13.5968895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5086481, upper bound: 10.5098262
time: 22.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5095230, upper bound: 10.5089576
time: 21.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1878815, 18.1898041
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8896103, 11.8908119
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6240387, 10.6249542
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2691841, 16.2713432
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4938927, 14.4971352
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4176254, 15.4199600
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7507019, 16.7525139
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9973068, 16.9992981
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8966217, 17.8987045
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8976822, 17.8978577
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8948860, 19.8941765
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1825027, 17.1800346
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3780823, 24.3773651
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9601440, 20.9626884
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3649445, 29.3657608
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6434708, 18.6444969
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2882843, 19.2865639
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2968292, 23.2988510
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8392105, 18.8363876
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0559464, 11.0545750
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3693886, 13.3693600
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0321922, 16.0303497
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5864487, 13.5869980
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2652969, 13.2619190
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0801697, 15.0788994
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5523071, 17.5522385
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4463272, 25.4441681
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5120392, 13.5112152
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1738815, 14.1722336
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2778893, 11.2772293
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1527100, 20.1525269
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3140106, 14.3130112
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9492798, 18.9503860
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0162735, 21.0186615
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5275421, 17.5291519
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7986679, 18.7990837
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1162872, 20.1180954
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0281982, 26.0271301
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4571533, 24.4591141
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1364975, 23.1391792
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9714012, 12.9730263
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7994614, 18.7986832
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5988808, 13.5992661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 752

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1615

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5088018, upper bound: 10.4850329
time: 23.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5096793, upper bound: 10.4841490
time: 24.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1861954, 18.1914940
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8885651, 11.8918533
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6228523, 10.6261406
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2687416, 16.2717819
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4925117, 14.4985123
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4171295, 15.4204559
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7511826, 16.7520294
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9964294, 17.0001831
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8940277, 17.9012909
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8951035, 17.9004288
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8940086, 19.8950577
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1841049, 17.1784325
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3790436, 24.3764038
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9604492, 20.9623871
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3643951, 29.3663025
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6428375, 18.6451263
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2851105, 19.2897301
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2968521, 23.2988281
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8392029, 18.8363953
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0558510, 11.0546722
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3693085, 13.3694382
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0321465, 16.0303917
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5870667, 13.5863838
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2660828, 13.2611313
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0811691, 15.0778961
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5533218, 17.5512199
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4461060, 25.4443817
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5125656, 13.5106888
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1739502, 14.1721649
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2788086, 11.2763062
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1544037, 20.1508408
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3143425, 14.3126755
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9500198, 18.9496460
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0164108, 21.0185242
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5288620, 17.5278358
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7992096, 18.7985458
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1179047, 20.1164780
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0284576, 26.0268707
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4594421, 24.4568253
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1365280, 23.1391487
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9716187, 12.9728088
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7994995, 18.7986450
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5990219, 13.5991287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1448

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 855

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4927182, upper bound: 10.4754178
time: 22.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5094033, upper bound: 10.4601199
time: 24.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1948700, 18.1947289
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8942490, 11.8945961
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6306915, 10.6307182
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2744293, 16.2737770
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5091782, 14.5086365
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4204941, 15.4210587
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7534866, 16.7539444
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0016174, 17.0017624
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9106216, 17.9109726
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9095764, 17.9092255
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8969727, 19.8969879
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1930962, 17.1929512
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3773727, 24.3764572
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9614868, 20.9592056
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3626328, 29.3636093
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6501770, 18.6504326
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2929878, 19.2926674
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2967606, 23.2977219
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8409271, 18.8418961
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0579205, 11.0582123
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3683167, 13.3692894
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0372658, 16.0374031
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5921707, 13.5925560
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2740097, 13.2738838
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0860062, 15.0868645
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5588455, 17.5597000
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4495850, 25.4505081
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5099487, 13.5113907
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1766739, 14.1778755
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2842827, 11.2856178
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1642914, 20.1657257
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3178749, 14.3180275
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9473953, 18.9459076
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0126953, 21.0103416
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5273819, 17.5256805
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7975540, 18.7955475
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1188507, 20.1174011
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0250702, 26.0223541
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4666977, 24.4649353
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1321564, 23.1283302
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9706764, 12.9689865
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7974014, 18.7962189
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6009369, 13.6005402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1285

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1629

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4724806, upper bound: 10.5067711
time: 23.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4912891, upper bound: 10.4879640
time: 18.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 43.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5085915, upper bound: 10.5044204
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5041071, upper bound: 10.5089053
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.4970247, upper bound: 10.5122285
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5107610, upper bound: 10.4984943
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.4932484, upper bound: 10.5028739
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5019958, upper bound: 10.4941247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5115864, upper bound: 10.5124460
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5115864, upper bound: 10.5124460
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5023236, upper bound: 10.5086125
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5069197, upper bound: 10.5040168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.4934527, upper bound: 10.4897464
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.4951433, upper bound: 10.4880521
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5095822, upper bound: 10.4987024
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.4984002, upper bound: 10.5098867
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5086481, upper bound: 10.5098262
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5095230, upper bound: 10.5089576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5088018, upper bound: 10.4850329
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5096793, upper bound: 10.4841490
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.4927182, upper bound: 10.4754178
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.5094033, upper bound: 10.4601199
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.4724806, upper bound: 10.5067711
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 43.26
Output dim: 18, lower bound: -10.4912891, upper bound: 10.4879640

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1878281, 18.1850853
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8879890, 11.8856201
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6190109, 10.6171875
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2907181, 16.2904739
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5193253, 14.5195198
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4143982, 15.4123802
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7278061, 16.7297440
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0009384, 17.0001831
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9003983, 17.8980103
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9183273, 17.9176483
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8968468, 19.8964500
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1580200, 17.1565399
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3644867, 24.3676300
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9061432, 20.9137115
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3502274, 29.3461914
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6483917, 18.6482391
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2833481, 19.2820930
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2909927, 23.2913742
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8445740, 18.8403282
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0483818, 11.0471325
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3596458, 13.3571873
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0023499, 15.9987183
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5810814, 13.5822563
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2594910, 13.2565498
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0715103, 15.0704689
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5503540, 17.5495644
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4677124, 25.4624863
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.4918404, 13.4898949
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1673012, 14.1656113
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2980766, 11.2967911
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1593704, 20.1596451
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3037186, 14.3040237
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.8970337, 18.9036674
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.8747368, 20.8890686
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4231873, 17.4334793
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7220078, 18.7296524
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0798264, 20.0846863
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -25.9554138, 25.9636002
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4406204, 24.4429169
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0192719, 23.0333328
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.8990173, 12.9075890
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7660370, 18.7698441
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5806274, 13.5805435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1614

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1716

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5079069, upper bound: 10.5000054
time: 20.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5041702, upper bound: 10.5037390
time: 30.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1866150, 18.1862984
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8860970, 11.8875122
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6175423, 10.6186562
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2915115, 16.2896767
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5194702, 14.5193710
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4128036, 15.4139786
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7299500, 16.7276077
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0004654, 17.0006561
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8987045, 17.8997040
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9183655, 17.9176102
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8966331, 19.8966599
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1555481, 17.1590080
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3673553, 24.3647614
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9125443, 20.9073105
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3468552, 29.3495560
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6486206, 18.6480141
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2821350, 19.2833061
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2918549, 23.2905121
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8399658, 18.8449364
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0466652, 11.0488510
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3570595, 13.3597755
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -15.9986115, 16.0024529
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5821419, 13.5811920
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2563553, 13.2596855
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0701904, 15.0717926
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5492706, 17.5506516
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4627762, 25.4674225
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.4891548, 13.4925842
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1654167, 14.1674957
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2967911, 11.2980728
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1593628, 20.1596527
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3029060, 14.3048401
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9028549, 18.8978462
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.8873634, 20.8764381
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4332123, 17.4234581
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7292557, 18.7224007
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0842590, 20.0802536
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -25.9621582, 25.9568481
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4422836, 24.4412613
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0318298, 23.0207748
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9066811, 12.8999214
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7694702, 18.7664108
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5819054, 13.5792656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1417

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 559

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4967196, upper bound: 10.5014890
time: 23.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4967196, upper bound: 10.5014890
time: 23.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1937637, 18.1919441
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8966599, 11.8959312
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6244392, 10.6237202
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2784729, 16.2775955
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5010223, 14.4990463
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4238205, 15.4234276
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7481537, 16.7459373
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0002136, 16.9991150
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9084702, 17.9065475
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9165878, 17.9155045
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8989792, 19.8988380
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1868286, 17.1898270
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3880539, 24.3880157
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9622955, 20.9580307
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3709106, 29.3712158
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6551361, 18.6542816
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2927094, 19.2924194
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.3005142, 23.2999725
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8312073, 18.8344269
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0527191, 11.0542316
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3720131, 13.3726845
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0360107, 16.0386276
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5869522, 13.5870972
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2578888, 13.2615967
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0738068, 15.0749969
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5506439, 17.5513535
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4475403, 25.4510040
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5169258, 13.5182915
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1682281, 14.1699257
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2788963, 11.2802963
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1591110, 20.1600647
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3085213, 14.3093834
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9406662, 18.9386292
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0044479, 20.9988747
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5174103, 17.5131607
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7935028, 18.7908211
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1235275, 20.1211624
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0181732, 26.0158310
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4602661, 24.4579239
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1491547, 23.1425133
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9494820, 12.9453506
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7927704, 18.7913818
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5918236, 13.5908508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1511

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 923

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4912095, upper bound: 10.5074887
time: 18.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4922485, upper bound: 10.5064287
time: 27.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1917801, 18.1939278
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8957596, 11.8968315
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6236076, 10.6245518
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2771606, 16.2789078
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4985886, 14.5014763
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4231186, 15.4241371
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7481689, 16.7459221
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9989624, 17.0003662
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9067230, 17.9082947
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9149017, 17.9171944
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8989639, 19.8988571
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1887817, 17.1878662
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3888931, 24.3871765
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9597244, 20.9606018
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3709106, 29.3712158
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6540909, 18.6553230
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2921448, 19.2929916
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.3003006, 23.3001862
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8330536, 18.8325806
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0535202, 11.0534344
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3726501, 13.3720474
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0375214, 16.0371208
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5871353, 13.5869102
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2596588, 13.2598228
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0749359, 15.0738716
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5515366, 17.5504532
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4490585, 25.4494781
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5180626, 13.5171585
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1694870, 14.1686668
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2802467, 11.2789421
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1604004, 20.1587830
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3095169, 14.3083878
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9400024, 18.9392853
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0014267, 21.0018997
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5160828, 17.5144844
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7924805, 18.7918434
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1225204, 20.1221771
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0164490, 26.0175629
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4592819, 24.4589081
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1450577, 23.1466141
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9471893, 12.9476471
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7922592, 18.7918930
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5920181, 13.5906525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5100791, upper bound: 10.4983609
time: 21.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5106275, upper bound: 10.4978128
time: 25.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1887932, 18.1865692
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8964653, 11.8951912
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6252899, 10.6237164
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2690277, 16.2676544
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5060692, 14.5045662
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4167061, 15.4151268
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7425919, 16.7442627
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0022964, 17.0014648
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9066086, 17.9041443
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9119415, 17.9119759
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.9003830, 19.8986855
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1907654, 17.1915436
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3809738, 24.3822784
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9712448, 20.9712067
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3560333, 29.3529434
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6409531, 18.6394806
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2922897, 19.2930107
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2921677, 23.2892380
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8466187, 18.8470001
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0589943, 11.0586185
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3723221, 13.3724060
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0368080, 16.0372429
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5882607, 13.5879097
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2750130, 13.2758732
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0902405, 15.0899429
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5586243, 17.5586777
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4524765, 25.4525604
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5220566, 13.5222893
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1798439, 14.1799774
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2833977, 11.2832031
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1627426, 20.1629410
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3205032, 14.3205566
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9449921, 18.9468307
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9990311, 21.0008240
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5271988, 17.5272560
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7967377, 18.7963829
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1255493, 20.1264954
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0110703, 26.0139465
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4735870, 24.4744949
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1401443, 23.1406174
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9572334, 12.9598122
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7916870, 18.7933731
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5946083, 13.5964241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 605

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 895

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5115187, upper bound: 10.5110263
time: 25.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5101661, upper bound: 10.5123781
time: 27.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1888008, 18.1865730
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8964958, 11.8951874
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6252975, 10.6237164
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2690659, 16.2676468
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5062065, 14.5044632
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4167213, 15.4151230
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7425766, 16.7442932
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0023346, 17.0014420
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9066772, 17.9041138
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9120026, 17.9119644
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.9004288, 19.8986702
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1906433, 17.1916008
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3809433, 24.3823090
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9712677, 20.9711761
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3560028, 29.3529510
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6410141, 18.6394501
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2923203, 19.2930222
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2922363, 23.2891846
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8465881, 18.8470535
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0590019, 11.0586243
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3723412, 13.3724174
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0367928, 16.0372620
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5883102, 13.5878639
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2749062, 13.2759762
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0902405, 15.0899849
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5586395, 17.5586929
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4524307, 25.4526138
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5220108, 13.5223312
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1798325, 14.1799927
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2834015, 11.2832031
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1627579, 20.1629333
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3205185, 14.3205719
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9449844, 18.9468536
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9990768, 21.0007858
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5272446, 17.5271912
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7967682, 18.7963486
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1255341, 20.1265259
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0110703, 26.0139618
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4735184, 24.4746246
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1401978, 23.1405640
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9572487, 12.9598389
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7916183, 18.7934341
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5944405, 13.5965424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1680

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 889

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4999194, upper bound: 10.5122951
time: 23.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5114357, upper bound: 10.5007823
time: 30.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1837387, 18.1792450
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8941765, 11.8922615
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6207314, 10.6188793
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2749176, 16.2738457
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4933891, 14.4915314
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4148979, 15.4137459
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7587814, 16.7603989
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0001755, 16.9991760
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8897858, 17.8854446
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9000092, 17.8967857
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8833694, 19.8809280
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1941414, 17.1949348
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3669128, 24.3685379
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9355469, 20.9361763
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3689575, 29.3683929
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6454926, 18.6433640
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2868576, 19.2850990
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.3011703, 23.3014069
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8410950, 18.8418388
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0629749, 11.0635223
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3717575, 13.3726234
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0434723, 16.0441246
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5795441, 13.5811462
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2712440, 13.2706261
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0853043, 15.0861397
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5519333, 17.5531998
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4459152, 25.4457397
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5174942, 13.5187149
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1804466, 14.1808205
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2699471, 11.2709198
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1555710, 20.1580429
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3143082, 14.3162880
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9479218, 18.9494629
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9833031, 20.9833946
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5268021, 17.5261154
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7793655, 18.7795410
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0990219, 20.1000366
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0110931, 26.0083923
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4326019, 24.4320526
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1119385, 23.1131058
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9669189, 12.9657211
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7942505, 18.7931900
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5914001, 13.5908318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1576

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4970681, upper bound: 10.5084749
time: 22.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5021859, upper bound: 10.5033483
time: 20.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1706848, 18.1766548
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8876419, 11.8903503
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6139183, 10.6173668
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2557373, 16.2589035
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4954300, 14.4981461
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4005814, 15.4047699
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7494278, 16.7458496
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9920425, 16.9945145
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8822556, 17.8899155
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9055328, 17.9058533
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8918381, 19.8961639
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1811981, 17.1821899
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3778763, 24.3716812
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9710388, 20.9654617
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3503418, 29.3534470
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6329651, 18.6363144
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2976151, 19.2978554
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2929688, 23.2945557
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8405151, 18.8419266
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0559158, 11.0555916
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3731155, 13.3734207
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0343170, 16.0331268
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5908661, 13.5893784
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2667198, 13.2691956
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0824776, 15.0835838
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5550919, 17.5543823
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4493637, 25.4513245
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5197372, 13.5210419
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1741409, 14.1742249
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2803154, 11.2807732
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1554031, 20.1542740
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3204269, 14.3187561
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9457932, 18.9405365
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0003357, 20.9916000
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5274048, 17.5252266
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7907715, 18.7853737
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1245880, 20.1197891
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0094910, 26.0062408
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4788666, 24.4791336
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1395721, 23.1310577
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9636078, 12.9602890
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7954941, 18.7942200
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5891075, 13.5878487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 923

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5094500, upper bound: 10.4951490
time: 20.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5060286, upper bound: 10.4985702
time: 29.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1723938, 18.1749535
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8884544, 11.8895416
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6149788, 10.6163101
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2572327, 16.2574081
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4965439, 14.4970284
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4022293, 15.4031181
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7500229, 16.7452583
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9931412, 16.9934158
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8844986, 17.8876724
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9064026, 17.9049835
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8921051, 19.8958969
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1800842, 17.1832962
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3769836, 24.3725662
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9713745, 20.9651260
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3512573, 29.3525314
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6335449, 18.6357269
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2983475, 19.2971268
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2935333, 23.2939911
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8397369, 18.8427124
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0553017, 11.0562057
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3732262, 13.3733101
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0337372, 16.0337143
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5902710, 13.5899773
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2659760, 13.2699413
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0816002, 15.0844574
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5546112, 17.5548630
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4489670, 25.4517212
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5195541, 13.5212326
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1733360, 14.1750259
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2796822, 11.2814064
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1543579, 20.1553192
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3198967, 14.3192863
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9456558, 18.9406662
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0002136, 20.9917259
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5271301, 17.5255089
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7899628, 18.7861862
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1244049, 20.1199722
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0087433, 26.0069885
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4794006, 24.4786072
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1398010, 23.1308289
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9641609, 12.9597359
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7954865, 18.7942276
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5892868, 13.5876732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1582

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4981520, upper bound: 10.5095707
time: 24.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4980181, upper bound: 10.5096087
time: 18.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1733627, 18.1683846
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8913765, 11.8895454
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6139450, 10.6110325
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2678223, 16.2660484
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5028114, 14.4993324
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4146652, 15.4139748
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7257309, 16.7296371
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -17.0005875, 16.9990158
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8920441, 17.8895531
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9070892, 17.9055099
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8933945, 19.8935165
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1889420, 17.1910782
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3696671, 24.3705521
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9648514, 20.9599304
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3450470, 29.3412552
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6336136, 18.6315651
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2923965, 19.2926025
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2855530, 23.2823639
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8445816, 18.8467445
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0583687, 11.0580750
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3645744, 13.3666649
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0300827, 16.0324631
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5867424, 13.5872650
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2667694, 13.2692318
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0874252, 15.0875969
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5576248, 17.5597420
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4510956, 25.4527359
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5202904, 13.5213203
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1794930, 14.1807861
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2822456, 11.2839508
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1505432, 20.1536942
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3192978, 14.3196602
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9405289, 18.9399529
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9990158, 20.9943237
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5271187, 17.5262222
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7953415, 18.7915077
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1229401, 20.1203232
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0095215, 26.0092392
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4745712, 24.4724274
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1389008, 23.1308594
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9594994, 12.9584656
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7940750, 18.7951584
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5934105, 13.5943336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1292

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1680

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5080255, upper bound: 10.5097759
time: 21.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5085937, upper bound: 10.5091674
time: 26.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1689453, 18.1728020
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8900223, 11.8908958
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6107101, 10.6142635
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2674026, 16.2664680
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.5000038, 14.5021439
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4145126, 15.4141273
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7306366, 16.7247353
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9992599, 17.0003433
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8883209, 17.8932686
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.9059677, 17.9066391
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8926163, 19.8942947
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1889877, 17.1910324
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3728714, 24.3673477
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9656219, 20.9591560
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3430328, 29.3432693
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6316452, 18.6335373
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2917862, 19.2932129
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2840271, 23.2838974
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8446350, 18.8466911
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0575523, 11.0588875
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3664894, 13.3647480
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0321884, 16.0303650
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5880775, 13.5859261
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2661324, 13.2698708
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0863724, 15.0886459
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5599670, 17.5573997
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4504395, 25.4533844
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5201607, 13.5214500
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1795807, 14.1806984
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2832146, 11.2829819
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1533127, 20.1509247
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3196564, 14.3192978
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9421692, 18.9383125
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9995499, 20.9937897
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5279274, 17.5254211
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7956390, 18.7912102
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1238098, 20.1194534
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0084076, 26.0103455
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4748383, 24.4721527
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1378098, 23.1319504
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9578743, 12.9600945
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7943344, 18.7948914
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5947685, 13.5929756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1412

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5069125, upper bound: 10.5076190
time: 24.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5081868, upper bound: 10.5063419
time: 19.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1857681, 18.1878090
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8888206, 11.8900566
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6211739, 10.6224346
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2697067, 16.2718925
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4924850, 14.4958916
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4172630, 15.4196167
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7465134, 16.7478027
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9974670, 16.9994659
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8906097, 17.8934326
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8987732, 17.8987961
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8890572, 19.8890839
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1817284, 17.1793785
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3737640, 24.3724594
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9499283, 20.9509621
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3629150, 29.3639603
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6400299, 18.6414871
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2868614, 19.2853012
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2985764, 23.3010101
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8387299, 18.8362846
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0563717, 11.0552082
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3700523, 13.3699226
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0337219, 16.0316086
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5866852, 13.5872116
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2625046, 13.2594872
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0787086, 15.0778351
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5523148, 17.5520325
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4462662, 25.4445801
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5111465, 13.5103912
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1729088, 14.1716614
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2762794, 11.2759819
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1513901, 20.1513977
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3147163, 14.3136063
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9437943, 18.9440613
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0085449, 21.0098457
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5267448, 17.5283775
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7940521, 18.7938232
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1078491, 20.1084747
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0259857, 26.0246277
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4475174, 24.4481354
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1248016, 23.1257057
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9710770, 12.9726830
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7985992, 18.7977829
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5984383, 13.5986938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1732

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5002908, upper bound: 10.4734847
time: 24.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4972753, upper bound: 10.4764984
time: 19.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1858902, 18.1876869
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8888512, 11.8900223
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6215172, 10.6220913
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2697372, 16.2718658
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4926453, 14.4957275
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4172859, 15.4195976
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7459869, 16.7483292
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9974747, 16.9994583
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8913498, 17.8926964
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8986130, 17.8989563
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8897972, 19.8883514
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1818504, 17.1792603
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3731689, 24.3730545
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9484177, 20.9524651
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3631287, 29.3637314
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6404572, 18.6410637
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2870216, 19.2851448
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2989883, 23.3005981
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8391037, 18.8359070
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0565815, 11.0550003
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3699493, 13.3700275
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0334473, 16.0318832
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5866623, 13.5872345
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2628670, 13.2591267
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0790977, 15.0774460
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5521011, 17.5522499
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4467392, 25.4441071
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5112152, 13.5103188
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1733131, 14.1712570
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2766380, 11.2756195
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1515732, 20.1512146
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3146057, 14.3137169
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9429550, 18.9449005
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -21.0074539, 21.0109367
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5267677, 17.5283585
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7934036, 18.7944717
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1066666, 20.1096649
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0256958, 26.0249023
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4461746, 24.4494705
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1230240, 23.1274872
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9710579, 12.9727020
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7985611, 18.7978134
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5983047, 13.5988274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 894

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1464

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5077525, upper bound: 10.4839997
time: 20.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5095272, upper bound: 10.4822183
time: 19.75 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 42.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5079069, upper bound: 10.5000054
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5041702, upper bound: 10.5037390
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4967196, upper bound: 10.5014890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4967196, upper bound: 10.5014890
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4912095, upper bound: 10.5074887
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4922485, upper bound: 10.5064287
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5100791, upper bound: 10.4983609
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5106275, upper bound: 10.4978128
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5115187, upper bound: 10.5110263
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5101661, upper bound: 10.5123781
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4999194, upper bound: 10.5122951
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5114357, upper bound: 10.5007823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4970681, upper bound: 10.5084749
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5021859, upper bound: 10.5033483
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5094500, upper bound: 10.4951490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5060286, upper bound: 10.4985702
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4981520, upper bound: 10.5095707
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4980181, upper bound: 10.5096087
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5080255, upper bound: 10.5097759
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5085937, upper bound: 10.5091674
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5069125, upper bound: 10.5076190
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5081868, upper bound: 10.5063419
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5002908, upper bound: 10.4734847
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.4972753, upper bound: 10.4764984
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5077525, upper bound: 10.4839997
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 42.23
Output dim: 18, lower bound: -10.5095272, upper bound: 10.4822183
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.23
Output dim: 18, lower bound: -10.5094033, upper bound: 10.4601199

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 35.11 + 1789.83 = 1824.94 seconds
