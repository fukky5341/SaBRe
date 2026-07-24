## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 1800 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.77 + 31.50 = 34.27 seconds
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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1629

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4975400, upper bound: 10.5164363
time: 17.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5164363, upper bound: 10.4975400
time: 20.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 38.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 38.06
Output dim: 18, lower bound: -10.4975400, upper bound: 10.5164363
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 38.06
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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4926917, upper bound: 10.5164252
time: 26.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4975287, upper bound: 10.5116005
time: 22.07 seconds

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5116004, upper bound: 10.4975288
time: 23.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5164251, upper bound: 10.4926917
time: 16.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 42.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 42.40
Output dim: 18, lower bound: -10.4926917, upper bound: 10.5164252
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 42.40
Output dim: 18, lower bound: -10.4975287, upper bound: 10.5116005
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 42.40
Output dim: 18, lower bound: -10.5116004, upper bound: 10.4975288
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 42.40
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

Time for backsubstitution: 2.15 seconds

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
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4911979, upper bound: 10.4948121
time: 20.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4710465, upper bound: 10.5149186
time: 17.21 seconds

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

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4960348, upper bound: 10.4899645
time: 24.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4759087, upper bound: 10.5100940
time: 22.51 seconds

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

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5100939, upper bound: 10.4759087
time: 29.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4899644, upper bound: 10.4960349
time: 16.76 seconds

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

Time for backsubstitution: 2.15 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5149185, upper bound: 10.4710465
time: 23.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4948121, upper bound: 10.4911979
time: 34.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 60.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 60.47
Output dim: 18, lower bound: -10.4911979, upper bound: 10.4948121
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 60.47
Output dim: 18, lower bound: -10.4710465, upper bound: 10.5149186
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 60.47
Output dim: 18, lower bound: -10.4960348, upper bound: 10.4899645
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 60.47
Output dim: 18, lower bound: -10.4759087, upper bound: 10.5100940
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 60.47
Output dim: 18, lower bound: -10.5100939, upper bound: 10.4759087
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 60.47
Output dim: 18, lower bound: -10.4899644, upper bound: 10.4960349
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 60.47
Output dim: 18, lower bound: -10.5149185, upper bound: 10.4710465
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 60.47
Output dim: 18, lower bound: -10.4948121, upper bound: 10.4911979

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1193466, 18.1131744
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8570290, 11.8525238
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6006355, 10.5964355
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2316360, 16.2320595
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4564209, 14.4509735
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3709068, 15.3683243
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7516174, 16.7552528
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9584656, 16.9534073
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8477631, 17.8402977
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8566513, 17.8522148
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8897209, 19.8901367
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1607475, 17.1629372
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3502960, 24.3549500
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9398041, 20.9388847
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3588486, 29.3533173
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6331940, 18.6333733
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2553787, 19.2508202
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2747574, 23.2750473
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8268051, 18.8244209
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0608559, 11.0583687
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3695946, 13.3670216
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0284424, 16.0251312
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5625076, 13.5640831
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2514267, 13.2531662
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0570030, 15.0584259
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5441208, 17.5445709
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4388962, 25.4351883
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5120621, 13.5110855
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1500053, 14.1492043
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2397118, 11.2415161
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1195297, 20.1232605
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3149109, 14.3145599
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9218445, 18.9275665
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9474487, 20.9569626
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4515915, 17.4637718
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7648315, 18.7702103
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1038818, 20.1060944
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0043335, 26.0096970
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4644012, 24.4661560
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0929108, 23.0968628
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9445724, 12.9503059
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7804337, 18.7850647
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6058426, 13.6113167

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4827377, upper bound: 10.4710893
time: 22.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4778201, upper bound: 10.4880599
time: 21.75 seconds

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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4625953, upper bound: 10.4912268
time: 21.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4576555, upper bound: 10.5081706
time: 24.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1132965, 18.1192284
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8536491, 11.8559036
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5977440, 10.5993271
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2284241, 16.2352676
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4530945, 14.4542999
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3664284, 15.3728065
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7520294, 16.7548409
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9547501, 16.9571228
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8411255, 17.8469315
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8534393, 17.8554268
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8888893, 19.8909760
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1631508, 17.1605339
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3539658, 24.3512802
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9398041, 20.9388847
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3577499, 29.3544159
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6315918, 18.6349716
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2522202, 19.2539787
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2756882, 23.2741165
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8271103, 18.8241158
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0611839, 11.0580425
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3689919, 13.3676224
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0281677, 16.0254097
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5644455, 13.5621452
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2528267, 13.2517643
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0593224, 15.0561028
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5450592, 17.5436325
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4386597, 25.4354248
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5125580, 13.5105896
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1518135, 14.1473961
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2417221, 11.2395020
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1223907, 20.1203995
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3165970, 14.3128738
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9237137, 18.9256935
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9509735, 20.9534454
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4556198, 17.4597435
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7682495, 18.7667885
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1054001, 20.1045685
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0054626, 26.0085602
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4639664, 24.4665909
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0942764, 23.0954971
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9450531, 12.9498253
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7813568, 18.7841339
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6058235, 13.6113396

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4876207, upper bound: 10.4662128
time: 22.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4826864, upper bound: 10.4831717
time: 25.68 seconds

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

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4674876, upper bound: 10.4863693
time: 27.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4625445, upper bound: 10.5032947
time: 25.55 seconds

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

Time for backsubstitution: 2.16 seconds

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
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5032947, upper bound: 10.4625445
time: 18.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4863693, upper bound: 10.4674877
time: 23.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1192322, 18.1132927
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8559074, 11.8536491
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5993271, 10.5977440
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2352676, 16.2284241
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4542999, 14.4530945
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3728065, 15.3664246
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7548447, 16.7520294
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9571228, 16.9547501
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8469315, 17.8411293
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8554230, 17.8534393
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8909721, 19.8888855
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1605339, 17.1631508
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3512802, 24.3539658
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9388733, 20.9398003
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3544083, 29.3577576
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6349716, 18.6315956
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2539825, 19.2522163
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2741165, 23.2756882
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8241119, 18.8271103
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0580406, 11.0611820
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3676224, 13.3689938
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0254059, 16.0281639
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5621414, 13.5644455
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2517624, 13.2528267
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0561028, 15.0593262
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5436325, 17.5450554
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4354248, 25.4386597
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5105896, 13.5125580
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1473961, 14.1518135
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2395020, 11.2417259
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1203995, 20.1223907
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3128738, 14.3165970
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9256897, 18.9237137
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9534454, 20.9509697
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4597397, 17.4556198
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7667923, 18.7682495
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1045685, 20.1054001
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0085602, 26.0054550
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4665909, 24.4639664
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0954971, 23.0942802
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9498253, 12.9450531
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7841339, 18.7813568
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6113396, 13.6058235

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4831716, upper bound: 10.4826865
time: 21.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4662128, upper bound: 10.4876207
time: 20.89 seconds

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

Time for backsubstitution: 2.15 seconds

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
time: 21.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4912267, upper bound: 10.4625953
time: 25.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1131744, 18.1193466
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8525238, 11.8570290
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5964355, 10.6006355
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2320557, 16.2316360
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4509735, 14.4564209
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3683205, 15.3709068
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7552490, 16.7516174
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9534073, 16.9584656
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8403015, 17.8477631
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8522186, 17.8566513
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8901405, 19.8897209
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1629372, 17.1607475
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3549500, 24.3502960
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9388885, 20.9398003
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3533096, 29.3588562
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6333694, 18.6331940
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2508163, 19.2553787
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2750473, 23.2747574
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8244171, 18.8268051
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0583687, 11.0608578
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3670235, 13.3695946
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0251312, 16.0284424
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5640869, 13.5625038
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2531662, 13.2514267
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0584221, 15.0570030
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5445709, 17.5441170
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4351883, 25.4388962
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5110855, 13.5120621
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1492043, 14.1500053
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2415161, 11.2397156
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1232605, 20.1195297
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3145599, 14.3149109
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9275665, 18.9218445
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9569626, 20.9474487
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4637680, 17.4515915
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7702103, 18.7648315
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1060944, 20.1038818
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0097046, 26.0043259
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4661560, 24.4644012
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0968628, 23.0929146
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9503059, 12.9445724
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7850647, 18.7804337
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6113205, 13.6058426

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4880599, upper bound: 10.4778202
time: 21.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4710892, upper bound: 10.4827378
time: 19.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 43.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4827377, upper bound: 10.4710893
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4778201, upper bound: 10.4880599
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4625953, upper bound: 10.4912268
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4576555, upper bound: 10.5081706
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4876207, upper bound: 10.4662128
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4826864, upper bound: 10.4831717
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4674876, upper bound: 10.4863693
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4625445, upper bound: 10.5032947
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.5032947, upper bound: 10.4625445
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4863693, upper bound: 10.4674877
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4831716, upper bound: 10.4826865
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4662128, upper bound: 10.4876207
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.5081705, upper bound: 10.4576556
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4912267, upper bound: 10.4625953
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4880599, upper bound: 10.4778202
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.34
Output dim: 18, lower bound: -10.4710892, upper bound: 10.4827378

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1180420, 18.1125145
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8564873, 11.8522682
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6001892, 10.5963192
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2283783, 16.2311249
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4527512, 14.4498062
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3686562, 15.3676262
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7493973, 16.7529526
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9565506, 16.9529495
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8464966, 17.8403282
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8525848, 17.8502045
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8894882, 19.8894043
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1587067, 17.1570358
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3501129, 24.3540421
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9317627, 20.9359474
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3582687, 29.3527298
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6330566, 18.6333542
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2528381, 19.2497368
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2746887, 23.2748795
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8248138, 18.8194237
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0609894, 11.0571785
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3695679, 13.3667278
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0274162, 16.0241127
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5623970, 13.5637131
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2495804, 13.2473984
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0549088, 15.0533867
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5437164, 17.5432129
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4374161, 25.4309387
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5105476, 13.5081329
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1486626, 14.1452217
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2381973, 11.2374687
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1182480, 20.1199341
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3144913, 14.3137589
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9184189, 18.9259033
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9384308, 20.9537811
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4514542, 17.4621773
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7615051, 18.7682991
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1000748, 20.1051407
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0020294, 26.0087051
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4628067, 24.4665833
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0806198, 23.0925522
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9389992, 12.9477234
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7796326, 18.7846603
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6056061, 13.6122246

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4726728, upper bound: 10.4704395
time: 19.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4820897, upper bound: 10.4610214
time: 21.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1189270, 18.1118660
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8568726, 11.8519821
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6005211, 10.5959911
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2308426, 16.2288017
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4552536, 14.4473038
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3702126, 15.3660774
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7494812, 16.7530289
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9580078, 16.9514999
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8478622, 17.8390312
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8552551, 17.8481483
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8889847, 19.8901253
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1548462, 17.1609383
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3494110, 24.3547745
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9371338, 20.9308548
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3582687, 29.3529510
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6331863, 18.6332397
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2545776, 19.2482796
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2745972, 23.2749939
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8218079, 18.8225632
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0596657, 11.0585690
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3693008, 13.3669987
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0274315, 16.0242195
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5621376, 13.5639915
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2456627, 13.2512283
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0519638, 15.0565720
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5427551, 17.5442009
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4346466, 25.4337463
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5091057, 13.5096016
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1460228, 14.1479378
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2356682, 11.2401924
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1162109, 20.1222000
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3141098, 14.3141899
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9206085, 18.9241409
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9441605, 20.9479446
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4502029, 17.4636345
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7629623, 18.7668839
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1029434, 20.1022873
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0035706, 26.0074005
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4649658, 24.4645615
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0886841, 23.0845642
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9426346, 12.9447365
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7801056, 18.7842636
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6068192, 13.6110802

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4677568, upper bound: 10.4874119
time: 21.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4771740, upper bound: 10.4779941
time: 21.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1203690, 18.1101837
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8566933, 11.8520584
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.6001892, 10.5963192
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2332153, 16.2262840
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4531708, 14.4493942
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3720436, 15.3642426
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7525711, 16.7497749
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9574738, 16.9520340
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8479385, 17.8388824
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8533249, 17.8494644
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8899460, 19.8889503
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1565704, 17.1591721
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3499756, 24.3541794
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9337997, 20.9339104
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3539352, 29.3570633
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6350098, 18.6314011
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2528076, 19.2497635
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2732391, 23.2763214
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8209686, 18.8232651
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0575371, 11.0606270
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3676033, 13.3686943
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0245323, 16.0270004
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5611839, 13.5649261
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2480621, 13.2489166
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0524521, 15.0558395
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5426941, 17.5442390
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4330521, 25.4353027
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5088921, 13.5097885
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1443977, 14.1494865
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2358856, 11.2397804
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1168213, 20.1213608
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3121796, 14.3160706
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9222717, 18.9220543
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9461517, 20.9460602
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4581680, 17.4554634
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7634506, 18.7663536
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1023636, 20.1028519
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0068665, 26.0038605
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4670181, 24.4623642
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0875854, 23.0855865
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9443855, 12.9423409
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7832413, 18.7810516
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6111870, 13.6066437

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4525282, upper bound: 10.4905823
time: 25.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4619442, upper bound: 10.4811627
time: 40.59 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4475951, upper bound: 10.5075250
time: 22.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4570072, upper bound: 10.4981059
time: 21.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1119843, 18.1185684
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8531075, 11.8556480
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5972977, 10.5992107
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2251663, 16.2343330
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4494324, 14.4531326
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3641777, 15.3721046
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7498093, 16.7525406
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9528351, 16.9566650
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8398590, 17.8469620
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8493729, 17.8534164
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8886642, 19.8902397
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1611023, 17.1546326
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3537903, 24.3503647
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9317703, 20.9359436
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3571701, 29.3538284
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6314621, 18.6349487
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2496796, 19.2528954
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2756195, 23.2739487
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8251190, 18.8191185
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0613136, 11.0568542
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3689690, 13.3673286
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0271416, 16.0243912
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5643425, 13.5617752
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2509804, 13.2459984
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0572281, 15.0510635
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5446548, 17.5422745
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4371796, 25.4311752
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5110435, 13.5076370
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1504669, 14.1434174
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2402077, 11.2354584
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1211014, 20.1170807
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3161812, 14.3120728
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9202881, 18.9240341
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9419479, 20.9502640
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4554825, 17.4581490
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7649231, 18.7648811
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1016006, 20.1036148
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0031586, 26.0075760
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4623718, 24.4670181
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0819855, 23.0911865
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9394836, 12.9472427
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7805557, 18.7837296
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6055832, 13.6122437

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4775569, upper bound: 10.4655635
time: 16.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4869745, upper bound: 10.4561484
time: 21.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1128693, 18.1179199
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8534927, 11.8553619
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5976295, 10.5988827
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2276306, 16.2320137
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4519348, 14.4506302
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3657265, 15.3705597
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7498856, 16.7526169
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9542923, 16.9552155
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8412247, 17.8456650
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8520508, 17.8513603
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8881607, 19.8909607
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1572418, 17.1585350
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3530807, 24.3510971
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9371338, 20.9308510
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3571701, 29.3540497
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6315918, 18.6348343
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2514191, 19.2514420
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2755280, 23.2740631
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8221130, 18.8222580
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0599899, 11.0582428
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3687019, 13.3675976
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0271416, 16.0244980
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5640755, 13.5620537
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2470627, 13.2498283
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0542831, 15.0542526
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5436935, 17.5432625
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4344177, 25.4339828
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5096092, 13.5091057
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1478310, 14.1461296
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2376785, 11.2381821
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1190643, 20.1193466
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3157959, 14.3125038
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9224777, 18.9222679
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9476776, 20.9444237
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4542313, 17.4596062
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7663803, 18.7634621
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1044693, 20.1007614
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0046997, 26.0062714
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4645309, 24.4649963
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0900497, 23.0832024
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9431152, 12.9442558
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7810364, 18.7833328
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6068001, 13.6110992

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4726244, upper bound: 10.4825232
time: 25.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4820412, upper bound: 10.4731077
time: 24.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1143188, 18.1162376
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8533134, 11.8554382
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5972977, 10.5992107
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2300110, 16.2294960
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4498444, 14.4527168
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3675652, 15.3687248
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7529831, 16.7493668
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9537582, 16.9557495
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8413086, 17.8455162
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8501129, 17.8526764
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8891068, 19.8897896
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1589661, 17.1567688
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3536530, 24.3505096
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9338074, 20.9339066
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3528366, 29.3581619
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6334152, 18.6329994
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2496490, 19.2529259
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2741699, 23.2753983
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8212738, 18.8229599
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0578613, 11.0603027
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3670006, 13.3692932
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0242424, 16.0272827
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5631294, 13.5629883
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2494659, 13.2475166
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0547791, 15.0535164
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5436325, 17.5433006
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4328156, 25.4355392
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5093880, 13.5092926
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1462059, 14.1476784
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2378960, 11.2377701
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1196823, 20.1184998
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3138657, 14.3143845
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9241409, 18.9201813
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9496689, 20.9425392
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4621964, 17.4514351
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7668686, 18.7629318
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1038818, 20.1013336
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0080109, 26.0027313
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4665833, 24.4628067
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0889511, 23.0842209
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9448662, 12.9418602
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7841644, 18.7801285
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6111679, 13.6066628

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4574206, upper bound: 10.4857231
time: 26.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4668387, upper bound: 10.4763051
time: 8.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1152039, 18.1155853
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8537025, 11.8551559
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5976295, 10.5988827
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2324677, 16.2271729
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4523468, 14.4502144
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3691139, 15.3671761
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7530670, 16.7494431
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9552078, 16.9542923
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8426743, 17.8442192
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8527908, 17.8506165
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8886032, 19.8905067
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1551056, 17.1606712
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3529434, 24.3512421
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9391708, 20.9288139
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3528366, 29.3583832
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6335373, 18.6328850
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2513885, 19.2514687
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2740784, 23.2755051
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8182678, 18.8260994
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0565414, 11.0616951
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3667336, 13.3695641
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0242577, 16.0273857
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5628624, 13.5632668
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2455444, 13.2513447
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0518341, 15.0567017
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5426712, 17.5442886
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4300461, 25.4383469
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5079460, 13.5107613
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1435661, 14.1503944
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2353668, 11.2404938
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1176453, 20.1207657
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3134804, 14.3148193
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9263306, 18.9184189
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9553986, 20.9367027
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4609451, 17.4528923
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7683258, 18.7615166
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1067581, 20.0984802
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0095520, 26.0014191
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4687424, 24.4607849
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0970154, 23.0762329
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9484978, 12.9388695
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7846375, 18.7797241
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6123810, 13.6055183

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4524771, upper bound: 10.5026472
time: 23.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4618943, upper bound: 10.4932321
time: 20.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1155853, 18.1152077
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8551559, 11.8537025
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5988808, 10.5976276
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2271729, 16.2324677
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4502106, 14.4523430
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3671761, 15.3691101
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7494431, 16.7530632
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9542923, 16.9552078
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8442154, 17.8426704
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8506165, 17.8527908
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8905106, 19.8886070
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1606750, 17.1551094
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3512421, 24.3529434
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9288177, 20.9391708
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3583755, 29.3528366
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6328812, 18.6335411
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2514648, 19.2513885
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2755127, 23.2740784
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8260956, 18.8182678
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0616913, 11.0565414
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3695641, 13.3667336
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0273857, 16.0242577
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5632668, 13.5628586
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2513428, 13.2455444
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0567017, 15.0518379
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5442886, 17.5426712
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4383469, 25.4300537
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5107613, 13.5079498
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1503944, 14.1435661
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2404938, 11.2353668
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1207657, 20.1176453
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3148193, 14.3134804
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9184189, 18.9263306
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9366989, 20.9553986
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4528885, 17.4609489
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7615128, 18.7683296
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0984802, 20.1067581
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0014191, 26.0095520
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4607849, 24.4687424
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0762329, 23.0970154
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9388695, 12.9484978
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7797241, 18.7846451
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6055222, 13.6123810

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4932321, upper bound: 10.4618944
time: 26.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5026472, upper bound: 10.4524772
time: 20.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1162415, 18.1143188
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8554382, 11.8533134
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5992126, 10.5972958
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2294922, 16.2300072
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4527130, 14.4498405
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3687248, 15.3675652
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7493668, 16.7529793
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9557495, 16.9537582
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8455124, 17.8413086
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8526764, 17.8501129
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8897934, 19.8891106
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1567688, 17.1589699
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3505096, 24.3536453
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9339066, 20.9338074
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3581619, 29.3528366
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6329956, 18.6334152
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2529297, 19.2496490
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2753983, 23.2741699
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8229599, 18.8212776
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0603027, 11.0578632
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3692932, 13.3670025
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0272789, 16.0242462
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5629921, 13.5631256
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2475166, 13.2494640
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0535126, 15.0547791
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5433044, 17.5436325
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4355392, 25.4328156
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5092964, 13.5093880
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1476784, 14.1462059
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2377701, 11.2378960
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1184998, 20.1196823
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3143845, 14.3138657
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9201813, 18.9241409
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9425430, 20.9496727
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4514389, 17.4621925
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7629318, 18.7668724
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1013336, 20.1038818
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0027313, 26.0080109
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4628067, 24.4665833
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0842209, 23.0889511
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9418602, 12.9448662
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7801285, 18.7841644
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6066628, 13.6111679

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4763050, upper bound: 10.4668388
time: 24.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4857230, upper bound: 10.4574207
time: 22.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1179199, 18.1128769
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8553619, 11.8534927
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5988808, 10.5976276
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2320099, 16.2276306
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4506302, 14.4519310
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3705635, 15.3657265
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7526169, 16.7498894
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9552155, 16.9542923
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8456650, 17.8412247
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8513641, 17.8520470
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8909531, 19.8881531
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1585388, 17.1572495
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3510971, 24.3530807
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9308472, 20.9371376
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3540421, 29.3571701
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6348343, 18.6315918
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2514343, 19.2514191
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2740631, 23.2755203
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8222580, 18.8221130
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0582428, 11.0599918
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3675957, 13.3687000
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0245018, 16.0271454
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5620537, 13.5640755
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2498283, 13.2470627
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0542526, 15.0542870
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5432587, 17.5436974
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4339828, 25.4344177
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5091057, 13.5096054
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1461296, 14.1478310
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2381821, 11.2376785
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1193466, 20.1190643
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3125038, 14.3157959
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9222717, 18.9224777
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9444199, 20.9476776
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4596024, 17.4542351
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7634659, 18.7663803
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1007690, 20.1044693
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0062714, 26.0047073
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4649963, 24.4645309
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0831985, 23.0900459
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9442558, 12.9431152
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7833328, 18.7810364
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6110992, 13.6067963

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4731077, upper bound: 10.4820412
time: 23.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4825231, upper bound: 10.4726244
time: 22.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1185684, 18.1119843
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8556480, 11.8531075
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5992126, 10.5972996
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2343369, 16.2251701
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4531326, 14.4494247
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3721046, 15.3641777
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7525406, 16.7498055
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9566650, 16.9528351
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8469620, 17.8398590
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8534164, 17.8493729
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8902359, 19.8886566
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1546326, 17.1611061
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3503723, 24.3537903
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9359436, 20.9317703
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3538284, 29.3571701
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6349487, 18.6314621
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2528992, 19.2496796
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2739487, 23.2756195
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8191147, 18.8251190
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0568542, 11.0613136
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3673286, 13.3689690
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0243950, 16.0271339
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5617714, 13.5643387
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2459984, 13.2509804
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0510635, 15.0572281
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5422745, 17.5446587
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4311752, 25.4371796
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5076408, 13.5110435
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1434174, 14.1504669
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2354584, 11.2402077
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1170807, 20.1211014
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3120728, 14.3161812
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9240341, 18.9202919
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9502640, 20.9419479
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4581528, 17.4554787
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7648773, 18.7649231
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1036148, 20.1015930
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0075684, 26.0031586
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4670181, 24.4623718
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0911865, 23.0819855
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9472427, 12.9394836
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7837296, 18.7805557
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6122437, 13.6055832

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4561484, upper bound: 10.4869746
time: 23.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4655634, upper bound: 10.4775570
time: 23.56 seconds

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4981058, upper bound: 10.4570073
time: 24.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5075250, upper bound: 10.4475951
time: 21.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1101837, 18.1203690
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8520584, 11.8566933
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5963211, 10.6001911
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2262878, 16.2332191
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4493942, 14.4531670
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3642464, 15.3720436
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7497787, 16.7525711
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9520340, 16.9574738
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8388824, 17.8479424
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8494644, 17.8533249
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8889542, 19.8899460
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1591797, 17.1565666
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3541794, 24.3499756
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9339142, 20.9338036
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3570633, 29.3539352
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6314011, 18.6350098
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2497711, 19.2528076
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2763290, 23.2732391
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8232651, 18.8209724
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0606270, 11.0575371
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3686943, 13.3676033
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0270042, 16.0245247
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5649300, 13.5611877
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2489166, 13.2480640
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0558395, 15.0524559
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5442352, 17.5426941
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4353104, 25.4330444
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5097923, 13.5088921
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1494865, 14.1443977
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2397804, 11.2358856
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1213608, 20.1168213
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3160706, 14.3121796
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9220505, 18.9222717
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9460602, 20.9461517
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4554672, 17.4581642
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7663498, 18.7634506
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1028519, 20.1023636
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0038605, 26.0068741
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4623642, 24.4670181
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0855865, 23.0875854
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9423409, 12.9443855
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7810516, 18.7832413
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6066437, 13.6111870

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4811626, upper bound: 10.4619443
time: 25.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4905822, upper bound: 10.4525282
time: 21.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1118622, 18.1189270
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8519821, 11.8568726
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5959892, 10.6005192
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2288055, 16.2308388
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4473038, 14.4552536
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3660774, 15.3702087
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7530289, 16.7494812
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9514999, 16.9580078
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8390274, 17.8478584
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8481445, 17.8552589
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8901291, 19.8889885
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1609344, 17.1548462
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3547745, 24.3494110
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9308548, 20.9371338
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3529434, 29.3582687
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6332397, 18.6331863
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2482758, 19.2545776
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2749939, 23.2745972
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8225632, 18.8218079
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0585670, 11.0596676
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3670006, 13.3693008
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0242119, 16.0274277
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5639915, 13.5621338
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2512283, 13.2456627
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0565720, 15.0519676
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5441971, 17.5427589
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4337463, 25.4346466
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5096016, 13.5091095
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1479378, 14.1460228
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2401924, 11.2356682
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1222000, 20.1162109
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3141899, 14.3141098
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9241409, 18.9206047
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9479446, 20.9441566
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4636307, 17.4502068
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7668839, 18.7629623
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1022873, 20.1029434
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0074005, 26.0035706
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4645615, 24.4649658
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0845642, 23.0886841
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9447365, 12.9426346
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7842636, 18.7801056
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6110802, 13.6068192

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4779941, upper bound: 10.4771740
time: 28.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4874118, upper bound: 10.4677568
time: 25.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.1125183, 18.1180382
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8522682, 11.8564873
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5963211, 10.6001911
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2311249, 16.2283783
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4498138, 14.4527512
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3676262, 15.3686600
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7529526, 16.7493973
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9529495, 16.9565506
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8403244, 17.8464966
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8502045, 17.8525848
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8894119, 19.8894920
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1570435, 17.1587029
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3540421, 24.3501129
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9359436, 20.9317703
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3527298, 29.3582687
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6333542, 18.6330605
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2497406, 19.2528381
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2748795, 23.2746887
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8194199, 18.8248138
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0571785, 11.0609894
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3667259, 13.3695679
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0241203, 16.0274124
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5637169, 13.5624008
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2473984, 13.2495804
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0533829, 15.0549088
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5432129, 17.5437202
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4309387, 25.4374161
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5081291, 13.5105476
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1452217, 14.1486626
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2374687, 11.2381973
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1199341, 20.1182480
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3137589, 14.3144913
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9259033, 18.9184189
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9537811, 20.9384270
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4621811, 17.4514503
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7683029, 18.7615051
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1051407, 20.1000748
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0086975, 26.0020294
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4665833, 24.4628067
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0925522, 23.0806198
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9477234, 12.9389992
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7846603, 18.7796326
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6122246, 13.6056061

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1783

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4610213, upper bound: 10.4820897
time: 22.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4704395, upper bound: 10.4726728
time: 21.43 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 46.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4726728, upper bound: 10.4704395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4820897, upper bound: 10.4610214
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4677568, upper bound: 10.4874119
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4771740, upper bound: 10.4779941
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4525282, upper bound: 10.4905823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4619442, upper bound: 10.4811627
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4475951, upper bound: 10.5075250
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4570072, upper bound: 10.4981059
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4775569, upper bound: 10.4655635
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4869745, upper bound: 10.4561484
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4726244, upper bound: 10.4825232
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4820412, upper bound: 10.4731077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4574206, upper bound: 10.4857231
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4668387, upper bound: 10.4763051
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4524771, upper bound: 10.5026472
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4618943, upper bound: 10.4932321
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4932321, upper bound: 10.4618944
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.5026472, upper bound: 10.4524772
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4763050, upper bound: 10.4668388
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4857230, upper bound: 10.4574207
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4731077, upper bound: 10.4820412
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4825231, upper bound: 10.4726244
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4561484, upper bound: 10.4869746
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4655634, upper bound: 10.4775570
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4981058, upper bound: 10.4570073
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.5075250, upper bound: 10.4475951
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4811626, upper bound: 10.4619443
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4905822, upper bound: 10.4525282
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4779941, upper bound: 10.4771740
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4874118, upper bound: 10.4677568
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4610213, upper bound: 10.4820897
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.53
Output dim: 18, lower bound: -10.4704395, upper bound: 10.4726728

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.0933800, 18.0848923
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8414764, 11.8356094
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5842628, 10.5786133
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2168045, 16.2181320
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4246140, 14.4186020
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3526573, 15.3497658
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7487488, 16.7519760
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9353027, 16.9293747
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8124542, 17.8026161
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8415298, 17.8377724
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8916245, 19.8915939
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1404419, 17.1409378
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3285599, 24.3347626
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9341888, 20.9381065
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3617554, 29.3560562
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6230164, 18.6220894
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2545204, 19.2507019
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2837830, 23.2828064
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8121262, 18.8082962
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0553741, 11.0525131
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3702736, 13.3672943
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0106888, 16.0099068
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5618095, 13.5635757
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2297173, 13.2297630
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0486832, 15.0478210
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5351105, 17.5355911
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4288025, 25.4232330
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5109215, 13.5083199
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1398315, 14.1372833
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2302475, 11.2304573
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1078262, 20.1106720
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3097458, 14.3100662
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9177475, 18.9252396
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9377365, 20.9530106
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4464302, 17.4572868
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7444229, 18.7529030
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0969543, 20.1022797
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -25.9959717, 26.0028229
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4641113, 24.4679337
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0778122, 23.0888062
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9346657, 12.9410324
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7803116, 18.7853546
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6005249, 13.6066704

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1645

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4670652, upper bound: 10.4697320
time: 24.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4707976, upper bound: 10.4597951
time: 20.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.0904121, 18.0878563
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8398285, 11.8372574
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5824852, 10.5803909
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2153854, 16.2195473
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4215469, 14.4216690
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3508034, 15.3516235
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7484207, 16.7523041
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9329758, 16.9317093
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8087845, 17.8062859
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8401489, 17.8391495
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8916855, 19.8915367
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1426010, 17.1387749
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3308411, 24.3324890
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9339294, 20.9383698
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3616028, 29.3562164
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6217957, 18.6233139
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2538033, 19.2514191
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2826157, 23.2839737
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8136902, 18.8067322
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0563240, 11.0515633
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3701324, 13.3674316
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0132065, 16.0073891
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5622597, 13.5631256
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2319450, 13.2275391
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0493393, 15.0471611
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5361023, 17.5345993
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4297028, 25.4223328
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5107384, 13.5085068
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1407242, 14.1363945
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2311897, 11.2295189
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1089783, 20.1095123
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3108025, 14.3090096
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9177551, 18.9252319
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9376526, 20.9530907
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4465675, 17.4571609
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7461090, 18.7512207
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0972137, 20.1020203
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -25.9961395, 26.0026550
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4641647, 24.4678802
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0768738, 23.0897446
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9323120, 12.9433861
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7803268, 18.7853394
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6000519, 13.6071434

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4764803, upper bound: 10.4603145
time: 21.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4802134, upper bound: 10.4503790
time: 33.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.0942726, 18.0842438
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8418655, 11.8353233
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5845909, 10.5782814
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2192612, 16.2158089
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4271164, 14.4160995
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3542061, 15.3482170
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7488327, 16.7520523
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9367599, 16.9279175
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8138199, 17.8013191
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8442001, 17.8357162
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8911209, 19.8923111
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1365891, 17.1448364
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3278580, 24.3354950
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9395599, 20.9330139
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3617554, 29.3562775
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6231461, 18.6219749
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2562599, 19.2492485
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2836914, 23.2829208
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8091125, 18.8114319
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0540543, 11.0539036
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3700027, 13.3675652
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0106964, 16.0100136
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5615425, 13.5638542
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2257996, 13.2335911
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0457382, 15.0510063
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5341492, 17.5365791
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4260406, 25.4260406
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5094795, 13.5097885
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1371956, 14.1399994
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2277184, 11.2331848
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1057816, 20.1129379
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3093605, 14.3104973
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9199295, 18.9234772
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9434662, 20.9471703
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4451942, 17.4587440
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7458801, 18.7514839
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0998230, 20.0994339
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -25.9975128, 26.0015106
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4662704, 24.4659195
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0858765, 23.0808220
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9382973, 12.9380455
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7807922, 18.7849579
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6017380, 13.6055260

Time for backsubstitution: 2.19 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4619955, upper bound: 10.4868131
time: 18.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4662069, upper bound: 10.4729994
time: 35.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.0913048, 18.0872040
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8402176, 11.8369751
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5828133, 10.5800629
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2178421, 16.2172241
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4240494, 14.4191666
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3523445, 15.3500786
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7485046, 16.7523842
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9344254, 16.9302521
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8101501, 17.8049889
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8428268, 17.8370934
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8911743, 19.8922577
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1387482, 17.1426773
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3301315, 24.3332214
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9392929, 20.9332771
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3616028, 29.3564377
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6219177, 18.6231995
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2555428, 19.2499619
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2825241, 23.2840881
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8106766, 18.8098717
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0550003, 11.0529537
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3698654, 13.3677025
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0132217, 16.0074921
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5620003, 13.5634003
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2280235, 13.2313652
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0463943, 15.0503502
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5351410, 17.5355873
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4269409, 25.4251404
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5092964, 13.5099716
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1380844, 14.1391068
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2286568, 11.2322426
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1069412, 20.1117783
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3104172, 14.3094406
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9199448, 18.9234695
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9433823, 20.9472542
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4453163, 17.4586182
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7475662, 18.7498016
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.1000900, 20.0991669
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -25.9976959, 26.0013504
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4663239, 24.4658661
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0849380, 23.0817604
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9359436, 12.9403992
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7807999, 18.7849426
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6012650, 13.6059990

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4714092, upper bound: 10.4773972
time: 16.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4756209, upper bound: 10.4635861
time: 18.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.0957069, 18.0825577
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8416862, 11.8353996
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5842628, 10.5786133
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2216415, 16.2132912
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4250259, 14.4181900
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3560371, 15.3463821
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7519226, 16.7488022
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9362259, 16.9284515
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8139038, 17.8011703
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8422699, 17.8370285
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8920822, 19.8911400
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1383057, 17.1430740
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3284225, 24.3348999
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9362259, 20.9360695
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3574219, 29.3603821
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6249695, 18.6201363
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2544899, 19.2507324
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2823334, 23.2842560
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8082809, 18.8121376
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0519257, 11.0559616
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3683052, 13.3692589
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0077972, 16.0127983
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5605965, 13.5647888
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2282028, 13.2312794
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0462265, 15.0502701
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5340805, 17.5366173
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4244385, 25.4275970
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5092659, 13.5099754
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1355667, 14.1415482
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2279358, 11.2327690
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1063995, 20.1120911
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3074303, 14.3123817
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9215927, 18.9213867
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9454575, 20.9452858
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4531441, 17.4505768
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7463684, 18.7509537
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0992432, 20.0999985
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0008240, 25.9979706
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4683228, 24.4637222
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0847855, 23.0818405
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9400482, 12.9356499
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7839203, 18.7817459
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6061058, 13.6010895

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4469250, upper bound: 10.4898692
time: 22.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4506570, upper bound: 10.4799432
time: 16.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -29.3796501, -2.3679805, -29.3796501, -2.3679805, -18.0927467, 18.0855217
1: -13.7514820, 2.7063797, -13.7514820, 2.7063797, -11.8400383, 11.8370514
2: -12.0407429, 4.0916910, -12.0407429, 4.0916910, -10.5824814, 10.5803947
3: -21.1549931, -0.8516860, -21.1549931, -0.8516860, -16.2202225, 16.2147064
4: -19.4722176, 2.7198329, -19.4722176, 2.7198329, -14.4219589, 14.4212570
5: -15.5569496, 4.2567692, -15.5569496, 4.2567692, -15.3541756, 15.3482399
6: -21.6929855, -0.4277515, -21.6929855, -0.4277515, -16.7515945, 16.7491302
7: -18.7597351, 2.4546218, -18.7597351, 2.4546218, -16.9338913, 16.9307861
8: -28.8983116, -1.3739376, -28.8983116, -1.3739376, -17.8102264, 17.8048401
9: -19.1554909, 2.5118909, -19.1554909, 2.5118909, -17.8408890, 17.8384094
10: -16.8799801, 5.3853064, -16.8799801, 5.3853064, -19.8921356, 19.8910828
11: -2.7382908, 15.8394194, -2.7382908, 15.8394194, -17.1404648, 17.1409111
12: -17.3910828, 13.1805677, -17.3910828, 13.1805677, -24.3306961, 24.3326263
13: -30.4297905, -1.5703397, -30.4297905, -1.5703397, -20.9359589, 20.9363365
14: -34.1165543, 0.3350258, -34.1165543, 0.3350258, -29.3572693, 29.3605499
15: -15.3718367, 5.1991844, -15.3718367, 5.1991844, -18.6237488, 18.6213608
16: -15.5073223, 6.3156924, -15.5073223, 6.3156924, -19.2537727, 19.2514458
17: -23.0876236, 1.8254423, -23.0876236, 1.8254423, -23.2811737, 23.2854233
18: 1.7971625, 23.2872906, 1.7971625, 23.2872906, -18.8098450, 18.8105774
19: -0.8437676, 11.5624161, -0.8437676, 11.5624161, -11.0528717, 11.0550137
20: -4.4773703, 9.6008234, -4.4773703, 9.6008234, -13.3681679, 13.3693981
21: -1.4039884, 15.6128635, -1.4039884, 15.6128635, -16.0103226, 16.0102768
22: -3.1010692, 11.4950447, -3.1010692, 11.4950447, -13.5610466, 13.5643387
23: -1.3769855, 15.6565342, -1.3769855, 15.6565342, -13.2304268, 13.2290554
24: -1.9068527, 16.3150673, -1.9068527, 16.3150673, -15.0468826, 15.0496140
25: -2.7293167, 16.4236832, -2.7293167, 16.4236832, -17.5350723, 17.5356293
26: -5.4133592, 21.1900482, -5.4133592, 21.1900482, -25.4253387, 25.4266968
27: -0.4581509, 15.6227045, -0.4581509, 15.6227045, -13.5090828, 13.5101624
28: -1.4986019, 15.5117226, -1.4986019, 15.5117226, -14.1364594, 14.1406555
29: -2.0640771, 12.6893330, -2.0640771, 12.6893330, -11.2288780, 11.2318306
30: -8.1705284, 14.8355179, -8.1705284, 14.8355179, -20.1075516, 20.1109390
31: 0.5057044, 16.0298195, 0.5057044, 16.0298195, -14.3084869, 14.3113213
32: -22.0244751, 2.0266757, -22.0244751, 2.0266757, -18.9216080, 18.9213791
33: -39.7462540, -10.4457474, -39.7462540, -10.4457474, -20.9453735, 20.9453697
34: -33.4343758, -10.0708122, -33.4343758, -10.0708122, -17.4532814, 17.4504471
35: -24.0979843, -0.8027523, -24.0979843, -0.8027523, -18.7480545, 18.7492714
36: -20.8149548, 5.2961092, -20.8149548, 5.2961092, -20.0995026, 20.0997314
37: -32.3214760, -2.6716881, -32.3214760, -2.6716881, -26.0009918, 25.9978104
38: -28.7828674, 0.6446667, -28.7828674, 0.6446667, -24.4683762, 24.4636688
39: -44.0017014, -10.2404833, -44.0017014, -10.2404833, -23.0838394, 23.0827789
40: -31.3580551, -13.0247660, -31.3580551, -13.0247660, -12.9376945, 12.9380035
41: -19.8990593, 2.0608931, -19.8990593, 2.0608931, -18.7839279, 18.7817383
42: -20.1087418, -3.5267005, -20.1087418, -3.5267005, -13.6056328, 13.6015625

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1645

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4563408, upper bound: 10.4804502
time: 26.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4600729, upper bound: 10.4705268
time: 29.15 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 57.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4670652, upper bound: 10.4697320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4707976, upper bound: 10.4597951
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4764803, upper bound: 10.4603145
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4802134, upper bound: 10.4503790
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4619955, upper bound: 10.4868131
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4662069, upper bound: 10.4729994
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4714092, upper bound: 10.4773972
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4756209, upper bound: 10.4635861
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4469250, upper bound: 10.4898692
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4506570, upper bound: 10.4799432
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4563408, upper bound: 10.4804502
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 57.93
Output dim: 18, lower bound: -10.4600729, upper bound: 10.4705268
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4475951, upper bound: 10.5075250
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4570072, upper bound: 10.4981059
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4775569, upper bound: 10.4655635
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4869745, upper bound: 10.4561484
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4726244, upper bound: 10.4825232
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4820412, upper bound: 10.4731077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4574206, upper bound: 10.4857231
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4668387, upper bound: 10.4763051
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4524771, upper bound: 10.5026472
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4618943, upper bound: 10.4932321
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4932321, upper bound: 10.4618944
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.5026472, upper bound: 10.4524772
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4763050, upper bound: 10.4668388
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4857230, upper bound: 10.4574207
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4731077, upper bound: 10.4820412
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4825231, upper bound: 10.4726244
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4561484, upper bound: 10.4869746
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4655634, upper bound: 10.4775570
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4981058, upper bound: 10.4570073
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.5075250, upper bound: 10.4475951
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4811626, upper bound: 10.4619443
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4905822, upper bound: 10.4525282
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4779941, upper bound: 10.4771740
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4874118, upper bound: 10.4677568
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4610213, upper bound: 10.4820897
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 57.93
Output dim: 18, lower bound: -10.4704395, upper bound: 10.4726728

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 34.27 + 1785.02 = 1819.28 seconds
