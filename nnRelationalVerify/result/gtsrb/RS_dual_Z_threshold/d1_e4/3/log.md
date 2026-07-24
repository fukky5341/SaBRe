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
execution time: IAR + RelationalAnalysis = 2.79 + 30.32 = 33.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 18, lower bound: -10.5176063, upper bound: 10.5176063

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1629

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4975400, upper bound: 10.5164363
time: 17.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5164363, upper bound: 10.4975400
time: 19.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 36.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 36.86
Output dim: 18, lower bound: -10.4975400, upper bound: 10.5164363
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 36.86
Output dim: 18, lower bound: -10.5164363, upper bound: 10.4975400

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1865692, 18.1841240
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8915787, 11.8902473
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6236191, 10.6223106
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2730827, 16.2718773
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4962387, 14.4936981
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4180145, 15.4165268
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7603836, 16.7604332
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9946594, 16.9924011
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9107819, 17.9085083
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8939056, 17.8919373
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8939476, 19.8947449
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1760635, 17.1779900
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3895340, 24.3906555
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9485626, 20.9456100
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3686218, 29.3685226
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6489639, 18.6487923
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2834740, 19.2821045
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2906723, 23.2914734
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8384552, 18.8396034
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0594788, 11.0601139
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3723831, 13.3723755
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0350266, 16.0348816
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5823936, 13.5832520
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2646561, 13.2665138
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0779724, 15.0795250
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5535965, 17.5541382
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4466858, 25.4475784
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5184784, 13.5186615
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1658783, 14.1675339
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2671585, 11.2692604
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1418152, 20.1441116
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3159561, 14.3162346
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9563828, 18.9563789
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9985352, 20.9968109
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5174408, 17.5188828
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.8010025, 18.8010139
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1201172, 20.1185150
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0193176, 26.0186996
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4620056, 24.4599838
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1205444, 23.1161575
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9649353, 12.9648056
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8018265, 18.8019257
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6075096, 13.6074219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4926917, upper bound: 10.5164252
time: 25.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4975287, upper bound: 10.5116005
time: 21.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1841278, 18.1865768
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8902473, 11.8915787
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6223106, 10.6236191
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2718773, 16.2730827
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4936981, 14.4962349
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.4165268, 15.4180145
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7604370, 16.7603874
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9924011, 16.9946594
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.9085083, 17.9107819
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8919373, 17.8939056
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8947411, 19.8939438
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1779938, 17.1760674
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3906631, 24.3895340
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9456100, 20.9485626
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3685303, 29.3686295
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6487961, 18.6489677
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2821007, 19.2834702
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2914734, 23.2906723
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8395996, 18.8384476
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0601120, 11.0594788
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3723755, 13.3723831
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0348740, 16.0350227
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5832481, 13.5823975
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2665100, 13.2646580
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0795288, 15.0779762
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5541382, 17.5535965
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4475784, 25.4466858
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5186615, 13.5184784
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1675339, 14.1658783
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2692604, 11.2671585
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1441116, 20.1418152
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3162346, 14.3159561
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9563751, 18.9563789
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9968109, 20.9985352
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.5188828, 17.5174408
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.8010178, 18.8010025
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1185150, 20.1201172
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0187073, 26.0193100
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4599838, 24.4620056
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1161575, 23.1205406
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9648056, 12.9649353
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.8019257, 18.8018265
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6074257, 13.6075096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5116004, upper bound: 10.4975288
time: 22.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5164251, upper bound: 10.4926917
time: 16.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 41.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 41.30
Output dim: 18, lower bound: -10.4926917, upper bound: 10.5164252
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 41.30
Output dim: 18, lower bound: -10.4975287, upper bound: 10.5116005
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 41.30
Output dim: 18, lower bound: -10.5116004, upper bound: 10.4975288
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 41.30
Output dim: 18, lower bound: -10.5164251, upper bound: 10.4926917

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1305466, 18.1220398
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8585930, 11.8538780
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6007652, 10.5965672
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2481155, 16.2436981
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4612885, 14.4554329
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3825722, 15.3766098
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7552032, 16.7556572
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9646225, 16.9586563
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8565369, 17.8476219
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8604584, 17.8552780
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8928261, 19.8927917
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1682129, 17.1725426
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3545685, 24.3593674
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9485321, 20.9455833
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3732605, 29.3720551
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6387024, 18.6369324
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2555771, 19.2510490
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2816162, 23.2833481
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8347244, 18.8361816
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0592766, 11.0602379
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3735771, 13.3729706
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0344925, 16.0340652
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5653610, 13.5681534
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2523460, 13.2556038
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0635300, 15.0674019
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5470657, 17.5485497
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4491653, 25.4498291
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5173416, 13.5180206
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1520729, 14.1555328
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2481003, 11.2522125
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1249466, 20.1300964
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3146858, 14.3166466
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9331818, 18.9350548
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9737091, 20.9755020
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4786911, 17.4841576
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7724380, 18.7758713
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1058502, 20.1057739
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0083313, 26.0088501
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4689407, 24.4664764
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1152573, 23.1122360
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9586334, 12.9589844
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7913132, 18.7923355
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6143761, 13.6142693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4911979, upper bound: 10.4948121
time: 19.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4710465, upper bound: 10.5149186
time: 16.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1244888, 18.1280937
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8552132, 11.8572578
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5978737, 10.5994587
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2449036, 16.2469101
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4579620, 14.4587555
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3780937, 15.3810883
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7556076, 16.7552490
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9609070, 16.9623642
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8498993, 17.8542557
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8572464, 17.8584900
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8919945, 19.8936310
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1706238, 17.1701393
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3582458, 24.3556976
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9485321, 20.9455795
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3721619, 29.3731537
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6371078, 18.6385307
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2524185, 19.2542114
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2825470, 23.2824173
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8350296, 18.8358765
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0596008, 11.0599117
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3729782, 13.3735695
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0342102, 16.0343437
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5672989, 13.5662155
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2537460, 13.2542019
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0658569, 15.0650826
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5480042, 17.5476112
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4489288, 25.4500580
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5178375, 13.5175285
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1538773, 14.1537285
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2501106, 11.2502022
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1278000, 20.1272430
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3163719, 14.3149605
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9350586, 18.9331818
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9772339, 20.9719849
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4827194, 17.4801292
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7758560, 18.7724495
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1073761, 20.1042557
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0094604, 26.0077133
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4684982, 24.4669113
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1166229, 23.1108742
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9591141, 12.9585037
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7922363, 18.7914047
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6143570, 13.6142921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4960348, upper bound: 10.4899645
time: 23.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4759087, upper bound: 10.5100940
time: 21.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1280975, 18.1244926
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8572617, 11.8552094
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5994606, 10.5978756
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2469101, 16.2449036
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4587555, 14.4579659
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3810921, 15.3780937
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7552490, 16.7556076
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9623718, 16.9609070
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8542557, 17.8498993
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8584900, 17.8572464
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8936272, 19.8919945
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1701355, 17.1706200
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3556976, 24.3582458
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9455795, 20.9485359
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3731537, 29.3721619
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6385269, 18.6371078
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2542114, 19.2524185
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2824173, 23.2825470
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8358765, 18.8350296
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0599098, 11.0596008
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3735695, 13.3729763
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0343475, 16.0342102
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5662155, 13.5672989
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2542038, 13.2537479
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0650787, 15.0658531
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5476074, 17.5480042
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4500580, 25.4489365
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5175323, 13.5178375
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1537285, 14.1538773
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2502022, 11.2501106
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1272430, 20.1278000
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3149605, 14.3163719
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9331818, 18.9350586
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9719849, 20.9772301
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4801254, 17.4827156
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7724457, 18.7758598
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1042557, 20.1073761
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0077209, 26.0094604
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4669113, 24.4684982
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1108780, 23.1166229
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9585037, 12.9591141
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7914047, 18.7922363
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6142883, 13.6143532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5100939, upper bound: 10.4759087
time: 28.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4899644, upper bound: 10.4960349
time: 16.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1220398, 18.1305428
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8538780, 11.8585930
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5965691, 10.6007671
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2436981, 16.2481155
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4554291, 14.4612923
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3766060, 15.3825760
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7556610, 16.7551994
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9586487, 16.9646225
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8476181, 17.8565331
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8552780, 17.8604584
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8927956, 19.8928299
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1725464, 17.1682167
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3593674, 24.3545685
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9455795, 20.9485321
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3720551, 29.3732605
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6369324, 18.6387062
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2510529, 19.2555809
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2833481, 23.2816162
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8361816, 18.8347244
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0602379, 11.0592766
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3729706, 13.3735771
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0340652, 16.0344887
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5681534, 13.5653610
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2556000, 13.2523479
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0673981, 15.0635338
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5485458, 17.5470657
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4498215, 25.4491730
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5180206, 13.5173416
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1555328, 14.1520729
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2522125, 11.2481003
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1300964, 20.1249466
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3166466, 14.3146858
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9350586, 18.9331856
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9755020, 20.9737091
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4841537, 17.4786873
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7758713, 18.7724380
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1057739, 20.1058502
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0088501, 26.0083237
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4664764, 24.4689407
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1122360, 23.1152573
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9589844, 12.9586334
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7923355, 18.7913132
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6142693, 13.6143761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5149185, upper bound: 10.4710465
time: 23.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4948121, upper bound: 10.4911979
time: 33.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 59.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 59.24
Output dim: 18, lower bound: -10.4911979, upper bound: 10.4948121
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 59.24
Output dim: 18, lower bound: -10.4710465, upper bound: 10.5149186
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 59.24
Output dim: 18, lower bound: -10.4960348, upper bound: 10.4899645
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 59.24
Output dim: 18, lower bound: -10.4759087, upper bound: 10.5100940
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 59.24
Output dim: 18, lower bound: -10.5100939, upper bound: 10.4759087
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 59.24
Output dim: 18, lower bound: -10.4899644, upper bound: 10.4960349
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 59.24
Output dim: 18, lower bound: -10.5149185, upper bound: 10.4710465
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 59.24
Output dim: 18, lower bound: -10.4948121, upper bound: 10.4911979

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1216812, 18.1108398
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8572350, 11.8523178
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6006317, 10.5964355
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2364731, 16.2272186
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4568329, 14.4505615
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3742867, 15.3649406
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7547913, 16.7520752
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9593811, 16.9524918
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8492126, 17.8388519
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8573914, 17.8514748
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8901711, 19.8896866
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1586113, 17.1650734
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3501511, 24.3550873
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9418335, 20.9368477
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3545151, 29.3576431
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6351471, 18.6314201
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2553482, 19.2508469
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2733154, 23.2764893
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8229599, 18.8282623
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0574074, 11.0618191
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3676262, 13.3689880
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0255585, 16.0280190
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5612946, 13.5652962
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2499084, 13.2546806
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0545464, 15.0608749
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5430908, 17.5456009
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4345322, 25.4395523
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5104065, 13.5127411
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1457405, 14.1534691
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2374001, 11.2438278
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1181030, 20.1246796
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3125954, 14.3168716
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9256973, 18.9237137
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9551697, 20.9492416
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4583054, 17.4570618
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7667770, 18.7682610
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1061630, 20.1038055
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0091705, 26.0048447
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4686127, 24.4619446
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0998840, 23.0898933
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9499550, 12.9449234
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7840424, 18.7814560
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6114235, 13.6057358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4625953, upper bound: 10.4912268
time: 20.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4576555, upper bound: 10.5081706
time: 24.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1156235, 18.1168938
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8538551, 11.8556976
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5977402, 10.5993271
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2332611, 16.2304306
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4535065, 14.4538841
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3698082, 15.3694229
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7552032, 16.7516670
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9556656, 16.9562073
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8425751, 17.8454857
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8541794, 17.8546867
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8893394, 19.8905182
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1610146, 17.1626701
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3538284, 24.3514175
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9418335, 20.9368477
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3534164, 29.3587418
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6335449, 18.6330185
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2521896, 19.2540092
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2742462, 23.2755585
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8232651, 18.8279572
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0577354, 11.0614948
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3670273, 13.3695869
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0252686, 16.0282974
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5632324, 13.5633583
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2513123, 13.2532806
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0568733, 15.0585518
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5440292, 17.5446625
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4342957, 25.4397888
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5109024, 13.5122452
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1475487, 14.1516609
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2394104, 11.2418137
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1209641, 20.1218262
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3142815, 14.3151855
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9275665, 18.9218407
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9586945, 20.9457207
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4623337, 17.4530296
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7701950, 18.7648430
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1076889, 20.1022797
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0103149, 26.0037155
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4681778, 24.4623795
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.1012421, 23.0885315
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9504356, 12.9444427
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7849655, 18.7805252
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6114044, 13.6057549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4674876, upper bound: 10.4863693
time: 26.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4625445, upper bound: 10.5032947
time: 24.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1168976, 18.1156235
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8556976, 11.8538551
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5993271, 10.5977402
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2304306, 16.2332649
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4538879, 14.4535103
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3694191, 15.3698120
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7516708, 16.7552032
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9562073, 16.9556656
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8454895, 17.8425751
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8546829, 17.8541832
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8905220, 19.8893394
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1626701, 17.1610146
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3514175, 24.3538284
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9368439, 20.9418373
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3587418, 29.3534241
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6330185, 18.6335487
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2540054, 19.2521896
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2755661, 23.2742462
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8279572, 18.8232651
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0614967, 11.0577335
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3695869, 13.3670273
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0283051, 16.0252762
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5633545, 13.5632286
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2532806, 13.2513103
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0585518, 15.0568733
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5446625, 17.5440292
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4397888, 25.4342957
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5122452, 13.5109024
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1516609, 14.1475487
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2418137, 11.2394142
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1218262, 20.1209641
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3151855, 14.3142815
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9218369, 18.9275665
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9457245, 20.9586906
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4530334, 17.4623299
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7648392, 18.7701988
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1022797, 20.1076889
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0037231, 26.0103073
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4623795, 24.4681778
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0885315, 23.1012459
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9444427, 12.9504356
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7805252, 18.7849655
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6057587, 13.6114044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5032947, upper bound: 10.4625445
time: 17.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4863693, upper bound: 10.4674877
time: 23.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1108398, 18.1216812
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8523178, 11.8572350
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5964355, 10.6006317
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2272186, 16.2364731
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4505615, 14.4568329
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3649406, 15.3742905
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7520752, 16.7547951
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9524918, 16.9593811
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8388519, 17.8492088
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8514709, 17.8573914
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8896828, 19.8901749
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1650734, 17.1586113
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3550873, 24.3501587
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9368439, 20.9418373
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3576431, 29.3545227
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6314240, 18.6351433
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2508469, 19.2553482
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2764893, 23.2733154
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8282623, 18.8229599
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0618172, 11.0574093
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3689880, 13.3676262
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0280151, 16.0255547
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5653000, 13.5612907
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2546806, 13.2499104
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0608788, 15.0545540
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5456009, 17.5430908
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4395523, 25.4345322
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5127411, 13.5104065
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1534691, 14.1457405
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2438278, 11.2374039
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1246796, 20.1181030
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3168716, 14.3125954
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9237137, 18.9256935
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9492416, 20.9551697
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4570618, 17.4583015
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7682571, 18.7667770
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1038055, 20.1061630
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0048523, 26.0091705
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4619446, 24.4686127
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0898972, 23.0998840
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9449234, 12.9499550
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7814560, 18.7840424
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6057358, 13.6114235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5081705, upper bound: 10.4576556
time: 21.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4912267, upper bound: 10.4625953
time: 25.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 49.11 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 49.11
Output dim: 18, lower bound: -10.4625953, upper bound: 10.4912268
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 49.11
Output dim: 18, lower bound: -10.4576555, upper bound: 10.5081706
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 49.11
Output dim: 18, lower bound: -10.4674876, upper bound: 10.4863693
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 49.11
Output dim: 18, lower bound: -10.4625445, upper bound: 10.5032947
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 49.11
Output dim: 18, lower bound: -10.5032947, upper bound: 10.4625445
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 49.11
Output dim: 18, lower bound: -10.4863693, upper bound: 10.4674877
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 49.11
Output dim: 18, lower bound: -10.5081705, upper bound: 10.4576556
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 49.11
Output dim: 18, lower bound: -10.4912267, upper bound: 10.4625953

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1212616, 18.1095314
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8570824, 11.8517723
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6005211, 10.5959911
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2356796, 16.2239647
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4556732, 14.4468880
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3735924, 15.3626938
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7526550, 16.7498550
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9589233, 16.9505768
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8493042, 17.8375854
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8560028, 17.8474045
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8894424, 19.8896713
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1527100, 17.1630745
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3492737, 24.3549118
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9391708, 20.9288177
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3539352, 29.3572845
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6351395, 18.6312866
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2545471, 19.2483063
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2731476, 23.2764359
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8179626, 18.8264046
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0562172, 11.0620193
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3673325, 13.3689651
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0245323, 16.0271072
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5609245, 13.5652046
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2441444, 13.2527447
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0495148, 15.0590248
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5417328, 17.5452271
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4302826, 25.4381104
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5074577, 13.5112572
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1417618, 14.1522026
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2333565, 11.2425041
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1147842, 20.1236267
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3117943, 14.3165054
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9244537, 18.9202881
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9518814, 20.9402237
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4569168, 17.4569206
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7649078, 18.7649345
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1052322, 20.0999985
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0084229, 26.0025482
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4691772, 24.4603500
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0956497, 23.0775986
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9480171, 12.9393539
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7837143, 18.7806549
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6124001, 13.6054993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4475951, upper bound: 10.5075250
time: 21.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4570072, upper bound: 10.4981059
time: 20.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1095276, 18.1212616
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8517761, 11.8570824
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5959892, 10.6005192
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2239609, 16.2356758
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4468918, 14.4556694
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3626900, 15.3735924
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7498550, 16.7526550
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9505768, 16.9589233
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8375854, 17.8493042
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8474045, 17.8560028
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8896713, 19.8894424
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1630707, 17.1527100
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3549118, 24.3492661
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9288177, 20.9391708
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3572769, 29.3539352
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6312866, 18.6351395
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2483063, 19.2545471
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2764359, 23.2731476
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8264008, 18.8179626
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0620193, 11.0562172
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3689651, 13.3673325
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0271111, 16.0245361
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5652046, 13.5609207
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2527466, 13.2441444
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0590210, 15.0495148
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5452271, 17.5417290
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4381180, 25.4302826
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5112572, 13.5074539
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1522026, 14.1417618
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2425041, 11.2333565
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1236267, 20.1147842
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3165054, 14.3117943
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9202881, 18.9244576
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9402237, 20.9518814
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4569168, 17.4569206
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7649307, 18.7649078
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0999985, 20.1052322
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0025482, 26.0084229
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4603500, 24.4691772
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0775986, 23.0956497
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9393539, 12.9480171
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7806549, 18.7837143
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6054993, 13.6124001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4981058, upper bound: 10.4570073
time: 24.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5075250, upper bound: 10.4475951
time: 20.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 47.80 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.80
Output dim: 18, lower bound: -10.4475951, upper bound: 10.5075250
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 47.80
Output dim: 18, lower bound: -10.4570072, upper bound: 10.4981059
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 47.80
Output dim: 18, lower bound: -10.4981058, upper bound: 10.4570073
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.80
Output dim: 18, lower bound: -10.5075250, upper bound: 10.4475951

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.0966072, 18.0819092
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8420715, 11.8351173
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5845909, 10.5782852
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2240982, 16.2109680
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4275284, 14.4156876
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3575935, 15.3448334
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7520065, 16.7488785
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9376831, 16.9270020
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8152618, 17.7998734
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8449402, 17.8349724
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8915710, 19.8918610
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1344528, 17.1469727
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3277206, 24.3356323
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9415894, 20.9309769
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3574219, 29.3606110
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6250992, 18.6200218
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2562294, 19.2492752
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2822418, 23.2843704
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8052750, 18.8152771
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0506058, 11.0573540
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3680344, 13.3695297
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0078125, 16.0129013
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5603294, 13.5650673
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2242851, 13.2351074
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0432892, 15.0534554
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5331192, 17.5376053
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4216766, 25.4304047
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5078316, 13.5114441
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1329308, 14.1442604
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2254066, 11.2354965
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1043625, 20.1143570
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3070450, 14.3128128
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9237823, 18.9196243
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9511871, 20.9394493
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4519081, 17.4520340
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7478256, 18.7495384
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1021118, 20.0971451
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0023651, 25.9966583
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4704819, 24.4617081
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0928421, 23.0738525
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9436798, 12.9326630
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7843933, 18.7813492
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6073189, 13.5999451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1645

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4418335, upper bound: 10.5069179
time: 23.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4460469, upper bound: 10.4931205
time: 25.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.0819130, 18.0966034
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8351173, 11.8420715
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5782814, 10.5845909
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2109680, 16.2241020
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4156876, 14.4275284
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3448372, 15.3575897
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7488785, 16.7520103
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9270020, 16.9376831
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.7998734, 17.8152618
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8349762, 17.8449440
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8918610, 19.8915749
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1469727, 17.1344490
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3356323, 24.3277206
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9309769, 20.9415932
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3606110, 29.3574295
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6200180, 18.6250992
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2492714, 19.2562294
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2843704, 23.2822418
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8152771, 18.8052750
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0573540, 11.0506020
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3695297, 13.3680363
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0129013, 16.0078125
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5650673, 13.5603333
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2351074, 13.2242832
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0534592, 15.0432930
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5376053, 17.5331192
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4304047, 25.4216766
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5114479, 13.5078278
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1442642, 14.1329308
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2354965, 11.2254066
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1143570, 20.1043625
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3128128, 14.3070450
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9196243, 18.9237823
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9394455, 20.9511871
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4520302, 17.4519005
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7495346, 18.7478294
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0971451, 20.1021118
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -25.9966583, 26.0023727
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4617081, 24.4704819
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0738525, 23.0928421
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9326630, 12.9436798
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7813492, 18.7843933
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.5999451, 13.6073227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=148, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1645

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4931204, upper bound: 10.4460470
time: 26.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5069178, upper bound: 10.4418335
time: 19.47 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 48.58 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 48.58
Output dim: 18, lower bound: -10.4418335, upper bound: 10.5069179
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 48.58
Output dim: 18, lower bound: -10.4460469, upper bound: 10.4931205
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 48.58
Output dim: 18, lower bound: -10.4931204, upper bound: 10.4460470
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 48.58
Output dim: 18, lower bound: -10.5069178, upper bound: 10.4418335

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 33.11 + 704.17 = 737.28 seconds
