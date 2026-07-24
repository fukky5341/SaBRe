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
execution time: IAR + RelationalAnalysis = 2.75 + 30.24 = 32.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 18, lower bound: -10.5176063, upper bound: 10.5176063

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4928948, upper bound: 10.5140667
time: 21.80 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5160989, upper bound: 10.5160993
time: 20.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 42.24 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 42.24
Output dim: 18, lower bound: -10.4928948, upper bound: 10.5140667
IS_B2, status: Status.UNKNOWN, split count: 1, time: 42.24
Output dim: 18, lower bound: -10.5160989, upper bound: 10.5160993

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -29.3456249, -2.3777218, -29.3212452, -2.3847294, -18.1541138, 18.1342010
1: -13.7447424, 2.7025986, -13.7399054, 2.6999366, -11.8881149, 11.8850479
2: -12.0287380, 4.0865593, -12.0200615, 4.0828757, -10.6119118, 10.6057167
3: -21.1200085, -0.8613391, -21.0944576, -0.8682437, -16.2301636, 16.2107430
4: -19.4481277, 2.7122545, -19.4313202, 2.7068806, -14.4760628, 14.4611702
5: -15.5241718, 4.2477551, -15.5004539, 4.2413936, -15.3778229, 15.3601608
6: -21.6853123, -0.4426556, -21.6798401, -0.4533482, -16.7110062, 16.7217636
7: -18.7390289, 2.4473591, -18.7244091, 2.4421372, -16.9739532, 16.9642715
8: -28.8831196, -1.3775148, -28.8724747, -1.3800669, -17.8878555, 17.8800240
9: -19.1194401, 2.5004084, -19.0932732, 2.4921360, -17.8622131, 17.8425064
10: -16.8627148, 5.3741026, -16.8505096, 5.3660316, -19.8625298, 19.8580856
11: -2.7224462, 15.8027973, -2.7111146, 15.7773027, -17.1158752, 17.1351471
12: -17.3790855, 13.1249733, -17.3704071, 13.0845251, -24.2855072, 24.3185730
13: -30.4053497, -1.5872488, -30.3874607, -1.5994458, -20.9189606, 20.9126282
14: -34.0906219, 0.2827811, -34.0716896, 0.2459955, -29.2529602, 29.2742157
15: -15.3408775, 5.1873083, -15.3186750, 5.1788816, -18.6025238, 18.5900383
16: -15.4941597, 6.3077049, -15.4849110, 6.3019667, -19.2666130, 19.2641144
17: -23.0755577, 1.7552001, -23.0667324, 1.7052493, -23.1626205, 23.2075806
18: 1.8113050, 23.2518997, 1.8215570, 23.2271061, -18.7802429, 18.7942924
19: -0.8337603, 11.5590076, -0.8266978, 11.5565281, -11.0444584, 11.0403481
20: -4.4625025, 9.5874872, -4.4518509, 9.5780163, -13.3358612, 13.3338890
21: -1.3895230, 15.5958300, -1.3792295, 15.5836792, -15.9969482, 15.9970398
22: -3.0899014, 11.4870462, -3.0819280, 11.4813833, -13.5597878, 13.5573349
23: -1.3689761, 15.6543980, -1.3632669, 15.6528902, -13.2605324, 13.2580299
24: -1.8971596, 16.3024921, -1.8902588, 16.2933502, -15.0603256, 15.0623016
25: -2.7161498, 16.4174919, -2.7067685, 16.4130478, -17.5314713, 17.5281982
26: -5.3964763, 21.1494904, -5.3843389, 21.1198082, -25.3669510, 25.3841095
27: -0.4422579, 15.5956860, -0.4308677, 15.5759802, -13.4631958, 13.4725571
28: -1.4889598, 15.5026646, -1.4820790, 15.4965506, -14.1524544, 14.1521187
29: -2.0577085, 12.6746883, -2.0531573, 12.6639595, -11.2512856, 11.2587204
30: -8.1564083, 14.8040104, -8.1462660, 14.7826881, -20.0961151, 20.1100616
31: 0.5194664, 16.0232544, 0.5292230, 16.0185814, -14.2940331, 14.2888794
32: -22.0077553, 2.0162935, -21.9959068, 2.0087729, -18.9249725, 18.9214401
33: -39.7080612, -10.4601593, -39.6800804, -10.4705591, -20.9523735, 20.9312096
34: -33.4065018, -10.0804005, -33.3860664, -10.0871925, -17.4853859, 17.4731827
35: -24.0760918, -0.8120942, -24.0602589, -0.8187640, -18.7607498, 18.7535400
36: -20.8037491, 5.2873521, -20.7957840, 5.2812610, -20.0928116, 20.0970230
37: -32.3009262, -2.6876197, -32.2864227, -2.6988277, -25.9503784, 25.9629669
38: -28.7684383, 0.6321559, -28.7580566, 0.6231661, -24.4041443, 24.4116745
39: -43.9663925, -10.2503929, -43.9415436, -10.2574930, -23.1005020, 23.0834427
40: -31.3413429, -13.0332279, -31.3293991, -13.0389166, -12.9196625, 12.9235687
41: -19.8846970, 2.0515866, -19.8745766, 2.0449114, -18.7635345, 18.7607956
42: -20.0973625, -3.5358217, -20.0893097, -3.5424228, -13.5592270, 13.5613823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4761775, upper bound: 10.5109089
time: 33.93 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4897288, upper bound: 10.5109089
time: 25.57 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -29.3740978, -2.3698292, -29.3781204, -2.2991886, -18.2761536, 18.1995087
1: -13.7491512, 2.7053847, -13.7508774, 2.7292869, -11.9236069, 11.9004250
2: -12.0383110, 4.0903001, -12.0410233, 4.1238146, -10.6664581, 10.6307526
3: -21.1507854, -0.8548102, -21.1523895, -0.7773037, -16.3537598, 16.2655144
4: -19.4683609, 2.7185516, -19.4714546, 2.7703457, -14.5654449, 14.5070648
5: -15.5523891, 4.2545581, -15.5558128, 4.3208408, -15.4868393, 15.4151382
6: -21.6915474, -0.4377804, -21.7116566, -0.4288092, -16.7485657, 16.7947540
7: -18.7554951, 2.4529166, -18.7613945, 2.4725957, -17.0227814, 17.0042114
8: -28.8947105, -1.3752656, -28.8997326, -1.3671265, -17.9149857, 17.9092751
9: -19.1510773, 2.5083821, -19.1571579, 2.5907707, -17.9900360, 17.9103699
10: -16.8750572, 5.3817797, -16.8922043, 5.4264355, -19.9406090, 19.9090538
11: -2.7354178, 15.8362942, -2.8462257, 15.8383799, -17.1815491, 17.2961311
12: -17.3888378, 13.1729717, -17.4768791, 13.1859970, -24.3934097, 24.4727097
13: -30.4181633, -1.5751452, -30.4200153, -1.5102386, -21.0142670, 20.9557076
14: -34.1109238, 0.3292418, -34.2411499, 0.3355913, -29.3491211, 29.4922485
15: -15.3639622, 5.1965041, -15.3677902, 5.2575226, -18.6917114, 18.6380272
16: -15.5045757, 6.3128281, -15.5258217, 6.3410034, -19.3255386, 19.3179893
17: -23.0840206, 1.8167305, -23.2014313, 1.8210914, -23.2837753, 23.4063339
18: 1.8026648, 23.2831345, 1.7152677, 23.2891388, -18.8373413, 18.9235191
19: -0.8411636, 11.5619812, -0.8911114, 11.5635271, -11.0804253, 11.1009350
20: -4.4740124, 9.5985193, -4.5425515, 9.6017342, -13.3648300, 13.4316368
21: -1.4008512, 15.6115360, -1.4814963, 15.6140327, -16.0276871, 16.1092834
22: -3.0971920, 11.4936771, -3.1125739, 11.4991322, -13.5879517, 13.5943642
23: -1.3747120, 15.6559944, -1.4227824, 15.6570320, -13.2805405, 13.3210678
24: -1.9031072, 16.3130798, -1.9531789, 16.3155003, -15.0837440, 15.1326904
25: -2.7250676, 16.4226952, -2.7590098, 16.4233437, -17.5513153, 17.5866089
26: -5.4077749, 21.1849098, -5.5382943, 21.1919785, -25.4420471, 25.5749741
27: -0.4531560, 15.6136169, -0.5070195, 15.6138573, -13.5055428, 13.5606728
28: -1.4962897, 15.5093756, -1.5486660, 15.5102568, -14.1837616, 14.2212524
29: -2.0625851, 12.6853275, -2.1000764, 12.6869259, -11.2769127, 11.3216324
30: -8.1675377, 14.8324537, -8.2515755, 14.8353081, -20.1550903, 20.2428284
31: 0.5089359, 16.0292950, 0.4576993, 16.0325966, -14.3360214, 14.3485603
32: -22.0215416, 2.0236840, -22.0255013, 2.0626631, -18.9873886, 18.9513359
33: -39.7408600, -10.4495325, -39.7528152, -10.3584948, -21.0959015, 21.0018387
34: -33.4302101, -10.0739527, -33.4346008, -10.0148869, -17.5853577, 17.5160294
35: -24.0893993, -0.8060138, -24.0961571, -0.7781491, -18.8126526, 18.7965469
36: -20.8101883, 5.2943721, -20.8175735, 5.3094654, -20.1215897, 20.1416245
37: -32.3165932, -2.6760654, -32.3376503, -2.6628942, -25.9880295, 26.0781631
38: -28.7788715, 0.6408024, -28.7959595, 0.6616683, -24.4536362, 24.4984131
39: -43.9958801, -10.2427826, -44.0047684, -10.1494942, -23.2309494, 23.1417618
40: -31.3546829, -13.0269814, -31.3596535, -13.0164890, -12.9530945, 12.9875793
41: -19.8960190, 2.0587792, -19.8999729, 2.0858097, -18.8150864, 18.7972183
42: -20.1074181, -3.5293195, -20.1175461, -3.5068622, -13.5968361, 13.6336632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1690

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4994571, upper bound: 10.5129850
time: 19.88 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5129844, upper bound: 10.5129849
time: 24.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 46.32 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 46.32
Output dim: 18, lower bound: -10.4761775, upper bound: 10.5109089
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 46.32
Output dim: 18, lower bound: -10.4897288, upper bound: 10.5109089
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 46.32
Output dim: 18, lower bound: -10.4994571, upper bound: 10.5129850
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 46.32
Output dim: 18, lower bound: -10.5129844, upper bound: 10.5129849

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -29.3295383, -2.3799505, -29.2922993, -2.3886075, -18.1334038, 18.1020393
1: -13.7376719, 2.7009256, -13.7272892, 2.6969187, -11.8779144, 11.8708153
2: -12.0200634, 4.0851173, -12.0044022, 4.0803394, -10.6003189, 10.5878487
3: -21.1052170, -0.8642507, -21.0679054, -0.8734436, -16.2098351, 16.1807480
4: -19.4325371, 2.7102823, -19.4032021, 2.7033439, -14.4580841, 14.4332314
5: -15.5112782, 4.2457838, -15.4772110, 4.2378721, -15.3610840, 15.3344078
6: -21.6832237, -0.4465227, -21.6761227, -0.4601650, -16.7002029, 16.7132187
7: -18.7255440, 2.4458933, -18.7001324, 2.4395761, -16.9580231, 16.9393692
8: -28.8681107, -1.3792715, -28.8455639, -1.3832083, -17.8690338, 17.8503227
9: -19.1005535, 2.4977987, -19.0592041, 2.4874206, -17.8389587, 17.8058929
10: -16.8598690, 5.3719053, -16.8454800, 5.3620553, -19.8560181, 19.8514099
11: -2.7188058, 15.7857628, -2.7047598, 15.7472544, -17.0849991, 17.1142120
12: -17.3760509, 13.1105032, -17.3650131, 13.0584831, -24.2555389, 24.2977219
13: -30.3929176, -1.5908270, -30.3656464, -1.6056447, -20.8994751, 20.8856049
14: -34.0829735, 0.2816954, -34.0580482, 0.2440190, -29.2410889, 29.2569656
15: -15.3343763, 5.1844058, -15.3070068, 5.1737585, -18.5902252, 18.5747299
16: -15.4813080, 6.3064394, -15.4618874, 6.2996869, -19.2512970, 19.2399292
17: -23.0720043, 1.7493658, -23.0604134, 1.6949294, -23.1461945, 23.1940689
18: 1.8159518, 23.2404976, 1.8298345, 23.2065582, -18.7559586, 18.7757568
19: -0.8315105, 11.5572586, -0.8227243, 11.5536070, -11.0408478, 11.0356731
20: -4.4586344, 9.5865698, -4.4449139, 9.5764427, -13.3299522, 13.3259258
21: -1.3861866, 15.5948219, -1.3732414, 15.5819149, -15.9897919, 15.9881783
22: -3.0881369, 11.4847193, -3.0788074, 11.4772539, -13.5520020, 13.5505676
23: -1.3662381, 15.6446934, -1.3584385, 15.6363983, -13.2450600, 13.2455883
24: -1.8946018, 16.2905884, -1.8857360, 16.2719307, -15.0384293, 15.0468864
25: -2.7131577, 16.4124165, -2.7015219, 16.4041557, -17.5216751, 17.5187531
26: -5.3920174, 21.1411247, -5.3764195, 21.1050377, -25.3478012, 25.3679962
27: -0.4392457, 15.5887842, -0.4254909, 15.5635576, -13.4487419, 13.4607124
28: -1.4863577, 15.4987297, -1.4773865, 15.4894772, -14.1435394, 14.1437263
29: -2.0565405, 12.6655092, -2.0510652, 12.6474037, -11.2350006, 11.2481003
30: -8.1526909, 14.7944708, -8.1397858, 14.7655182, -20.0756989, 20.0938873
31: 0.5224724, 16.0212612, 0.5345974, 16.0150852, -14.2872391, 14.2806892
32: -22.0046425, 2.0137486, -21.9903030, 2.0044346, -18.9178009, 18.9137573
33: -39.7024765, -10.4632006, -39.6705704, -10.4759302, -20.9415665, 20.9190750
34: -33.4046402, -10.0827045, -33.3827515, -10.0913868, -17.4747238, 17.4650192
35: -24.0737743, -0.8150687, -24.0561981, -0.8240397, -18.7528839, 18.7461548
36: -20.8013992, 5.2848577, -20.7916145, 5.2767859, -20.0840302, 20.0888367
37: -32.2958908, -2.6911106, -32.2774506, -2.7050452, -25.9374390, 25.9498978
38: -28.7636986, 0.6259379, -28.7495384, 0.6122146, -24.3831253, 24.3947296
39: -43.9547882, -10.2521915, -43.9212418, -10.2606974, -23.0889816, 23.0677719
40: -31.3370399, -13.0336304, -31.3218346, -13.0396147, -12.9123306, 12.9130936
41: -19.8827057, 2.0468781, -19.8710556, 2.0365260, -18.7527466, 18.7523346
42: -20.0950928, -3.5391123, -20.0853310, -3.5482535, -13.5491028, 13.5534744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=146, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1693

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4689578, upper bound: 10.5081133
time: 25.18 seconds

## Relational analysis of IS_B1_B1_B2

### Relational analysis result of IS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4733922, upper bound: 10.5081133
time: 22.64 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -29.3439083, -2.3786278, -29.3207569, -2.3177471, -18.2199287, 18.1248245
1: -13.7435217, 2.7019198, -13.7391949, 2.7360513, -11.9235535, 11.8800507
2: -12.0277243, 4.0859804, -12.0193195, 4.1271172, -10.6550446, 10.6008301
3: -21.1181545, -0.8627229, -21.0937691, -0.8001575, -16.2980423, 16.2023163
4: -19.4465256, 2.7113018, -19.4324532, 2.7630267, -14.5302124, 14.4558372
5: -15.5224037, 4.2469459, -15.4996996, 4.3037071, -15.4392090, 15.3534660
6: -21.6844616, -0.4436216, -21.7008629, -0.4496641, -16.7107391, 16.7449837
7: -18.7371616, 2.4466257, -18.7232323, 2.4858789, -17.0162964, 16.9586029
8: -28.8806629, -1.3782778, -28.8721523, -1.3323355, -17.9328613, 17.8716774
9: -19.1175861, 2.4989061, -19.0957241, 2.5603566, -17.9290543, 17.8372040
10: -16.8619118, 5.3731203, -16.8861847, 5.3920345, -19.8847275, 19.8993835
11: -2.7204607, 15.8010101, -2.7747946, 15.7801285, -17.1111488, 17.1950188
12: -17.3782234, 13.1230373, -17.4397926, 13.0880260, -24.2866135, 24.3854218
13: -30.4030895, -1.5886540, -30.3892937, -1.5443382, -20.9723053, 20.9049416
14: -34.0882187, 0.2826324, -34.0991058, 0.2650805, -29.2667236, 29.3085938
15: -15.3395252, 5.1865349, -15.3227205, 5.2194929, -18.6408386, 18.5923119
16: -15.4916048, 6.3070636, -15.4920578, 6.3581104, -19.3198318, 19.2667503
17: -23.0738125, 1.7520356, -23.1007004, 1.7172234, -23.1691360, 23.2391586
18: 1.8133011, 23.2493591, 1.7453971, 23.2324696, -18.7798309, 18.8696289
19: -0.8329735, 11.5574989, -0.8632860, 11.5585232, -11.0525208, 11.0741444
20: -4.4614568, 9.5867834, -4.4901762, 9.5813065, -13.3426704, 13.3702297
21: -1.3886819, 15.5954208, -1.4093871, 15.5896873, -16.0054817, 16.0234299
22: -3.0893595, 11.4860506, -3.1069746, 11.4806271, -13.5634537, 13.5808907
23: -1.3677859, 15.6531267, -1.4124966, 15.6537428, -13.2570457, 13.3031349
24: -1.8955154, 16.2992935, -1.9442329, 16.2946548, -15.0569839, 15.1118011
25: -2.7147875, 16.4144077, -2.7525291, 16.4125156, -17.5326385, 17.5677261
26: -5.3941817, 21.1429863, -5.4691105, 21.1191101, -25.3657150, 25.4668503
27: -0.4409432, 15.5946789, -0.4738503, 15.5782614, -13.4610901, 13.5129013
28: -1.4881330, 15.5013809, -1.5224166, 15.4986296, -14.1572609, 14.1882248
29: -2.0570679, 12.6735945, -2.0839536, 12.6655464, -11.2491264, 11.2905693
30: -8.1544800, 14.8029308, -8.1880560, 14.7850790, -20.0938339, 20.1471252
31: 0.5201979, 16.0223236, 0.4841828, 16.0202942, -14.3021355, 14.3309326
32: -22.0045891, 2.0143671, -22.0077839, 2.0078564, -18.9226837, 18.9373856
33: -39.6989784, -10.4614258, -39.6884003, -10.4297733, -20.9822235, 20.9470253
34: -33.4061012, -10.0813131, -33.4156189, -10.0832472, -17.4870300, 17.5047913
35: -24.0718098, -0.8131442, -24.0679321, -0.7978172, -18.7778320, 18.7656860
36: -20.8026810, 5.2852044, -20.8093052, 5.2810354, -20.0887756, 20.1127014
37: -32.2982788, -2.6950564, -32.3142548, -2.7068648, -25.9447632, 25.9998322
38: -28.7669563, 0.6282115, -28.8031940, 0.6233902, -24.3971939, 24.4690704
39: -43.9621353, -10.2514467, -43.9561653, -10.2187195, -23.1313629, 23.1037140
40: -31.3392792, -13.0333691, -31.3424149, -13.0359344, -12.9182510, 12.9402504
41: -19.8838005, 2.0501854, -19.9013138, 2.0488343, -18.7669220, 18.7901154
42: -20.0963249, -3.5376964, -20.1175232, -3.5393119, -13.5591850, 13.5949249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=146, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1693

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4825028, upper bound: 10.5081133
time: 31.24 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4869362, upper bound: 10.5081133
time: 21.57 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -29.3579750, -2.3720045, -29.3491116, -2.3030949, -18.2554321, 18.1673622
1: -13.7421169, 2.7036977, -13.7382488, 2.7262635, -11.9133873, 11.8861656
2: -12.0296068, 4.0888872, -12.0253677, 4.1212931, -10.6548805, 10.6128654
3: -21.1359863, -0.8577290, -21.1258316, -0.7824917, -16.3334198, 16.2355118
4: -19.4527359, 2.7165680, -19.4433479, 2.7668552, -14.5474854, 14.4791412
5: -15.5395107, 4.2525663, -15.5325699, 4.3173265, -15.4701042, 15.3893852
6: -21.6894722, -0.4416447, -21.7079506, -0.4356031, -16.7377701, 16.7862244
7: -18.7419796, 2.4514699, -18.7370434, 2.4700708, -17.0068665, 16.9792786
8: -28.8796539, -1.3770003, -28.8727322, -1.3702722, -17.8961716, 17.8796120
9: -19.1322079, 2.5057254, -19.1231270, 2.5860946, -17.9667969, 17.8737717
10: -16.8722420, 5.3795319, -16.8871822, 5.4224887, -19.9340858, 19.9023399
11: -2.7317505, 15.8192797, -2.8399112, 15.8083391, -17.1506500, 17.2752228
12: -17.3858700, 13.1585274, -17.4714622, 13.1600609, -24.3634720, 24.4519119
13: -30.4057217, -1.5786939, -30.3981934, -1.5164976, -20.9947357, 20.9287033
14: -34.1032982, 0.3281288, -34.2276230, 0.3336034, -29.3372574, 29.4750061
15: -15.3574438, 5.1936178, -15.3561611, 5.2524605, -18.6794434, 18.6227417
16: -15.4917202, 6.3115492, -15.5027905, 6.3387427, -19.3102036, 19.2937737
17: -23.0804405, 1.8109517, -23.1951408, 1.8107471, -23.2673645, 23.3928146
18: 1.8073211, 23.2717037, 1.7235007, 23.2686005, -18.8130112, 18.9049759
19: -0.8389163, 11.5602427, -0.8871250, 11.5606213, -11.0768185, 11.0962830
20: -4.4701324, 9.5976028, -4.5356188, 9.6001616, -13.3588982, 13.4236984
21: -1.3974953, 15.6105366, -1.4755607, 15.6122704, -16.0205002, 16.1004791
22: -3.0954652, 11.4913416, -3.1094391, 11.4950199, -13.5801659, 13.5876045
23: -1.3720117, 15.6462793, -1.4179688, 15.6405439, -13.2650375, 13.3086700
24: -1.9004998, 16.3011856, -1.9486451, 16.2940826, -15.0618210, 15.1173058
25: -2.7220912, 16.4176369, -2.7537436, 16.4144077, -17.5414963, 17.5771713
26: -5.4032845, 21.1765499, -5.5303998, 21.1772079, -25.4228516, 25.5588379
27: -0.4501219, 15.6067142, -0.5016332, 15.6014280, -13.4910431, 13.5488243
28: -1.4936728, 15.5054531, -1.5439944, 15.5031605, -14.1748772, 14.2128868
29: -2.0614116, 12.6761312, -2.0980277, 12.6703529, -11.2606316, 11.3110123
30: -8.1638155, 14.8229342, -8.2450809, 14.8181562, -20.1346588, 20.2267075
31: 0.5119448, 16.0272961, 0.4630322, 16.0290985, -14.3292656, 14.3403778
32: -22.0183964, 2.0211315, -22.0198479, 2.0583262, -18.9802399, 18.9436417
33: -39.7353096, -10.4525633, -39.7432938, -10.3637867, -21.0851059, 20.9897079
34: -33.4283295, -10.0762930, -33.4312706, -10.0190296, -17.5746994, 17.5078316
35: -24.0870342, -0.8089776, -24.0921078, -0.7834277, -18.8048248, 18.7891731
36: -20.8078728, 5.2918797, -20.8133469, 5.3050385, -20.1128311, 20.1334152
37: -32.3115158, -2.6794996, -32.3286667, -2.6690760, -25.9750824, 26.0651855
38: -28.7741184, 0.6346469, -28.7874374, 0.6507502, -24.4326324, 24.4814911
39: -43.9842453, -10.2445936, -43.9843712, -10.1526861, -23.2194595, 23.1261063
40: -31.3503780, -13.0273838, -31.3520699, -13.0172014, -12.9457397, 12.9771385
41: -19.8940144, 2.0540881, -19.8964005, 2.0774381, -18.8043137, 18.7887802
42: -20.1051483, -3.5325761, -20.1135483, -3.5127130, -13.5867386, 13.6258011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=146, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1399

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1693

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4922088, upper bound: 10.5101926
time: 20.87 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4966684, upper bound: 10.5101926
time: 20.29 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -29.3723011, -2.3707008, -29.3775826, -2.2321653, -18.3419647, 18.1901398
1: -13.7479305, 2.7046893, -13.7501755, 2.7654226, -11.9590569, 11.8954353
2: -12.0372915, 4.0897589, -12.0402765, 4.1680794, -10.7096062, 10.6258392
3: -21.1489353, -0.8562179, -21.1517181, -0.7091813, -16.4216194, 16.2570953
4: -19.4667244, 2.7176375, -19.4726067, 2.8264437, -14.6196289, 14.5017586
5: -15.5506306, 4.2536907, -15.5550594, 4.3831840, -15.5482635, 15.4084206
6: -21.6907082, -0.4387417, -21.7327003, -0.4251156, -16.7483215, 16.8179550
7: -18.7536049, 2.4522095, -18.7602196, 2.5163989, -17.0651169, 16.9985428
8: -28.8922329, -1.3759966, -28.8994122, -1.3193502, -17.9600143, 17.9009743
9: -19.1492424, 2.5068865, -19.1596222, 2.6590102, -18.0568237, 17.9050140
10: -16.8742580, 5.3807611, -16.9278488, 5.4524913, -19.9627838, 19.9502602
11: -2.7334490, 15.8345165, -2.9098854, 15.8412046, -17.1768150, 17.3560181
12: -17.3879471, 13.1710587, -17.5463333, 13.1895227, -24.3945541, 24.5395889
13: -30.4158516, -1.5764914, -30.4218693, -1.4551620, -21.0675735, 20.9480171
14: -34.1085815, 0.3290257, -34.2687073, 0.3546667, -29.3628922, 29.5266647
15: -15.3626480, 5.1957154, -15.3719215, 5.2982411, -18.7300720, 18.6403046
16: -15.5020180, 6.3122044, -15.5329628, 6.3971477, -19.3787613, 19.3206444
17: -23.0823021, 1.8136084, -23.2354298, 1.8330326, -23.2902451, 23.4378967
18: 1.8046727, 23.2805977, 1.6390896, 23.2944736, -18.8368607, 18.9988480
19: -0.8403506, 11.5604830, -0.9277177, 11.5655174, -11.0884781, 11.1347733
20: -4.4729648, 9.5978212, -4.5808992, 9.6050110, -13.3716125, 13.4680443
21: -1.4000359, 15.6111584, -1.5117188, 15.6200428, -16.0362091, 16.1357079
22: -3.0966935, 11.4926662, -3.1376383, 11.4984417, -13.5916405, 13.6179123
23: -1.3735456, 15.6547413, -1.4720278, 15.6579247, -13.2770653, 13.3662529
24: -1.9014311, 16.3098755, -2.0071821, 16.3168221, -15.0803795, 15.1822281
25: -2.7237449, 16.4196167, -2.8048182, 16.4228077, -17.5524902, 17.6261711
26: -5.4054813, 21.1783867, -5.6231155, 21.1912403, -25.4407883, 25.6577225
27: -0.4518280, 15.6126308, -0.5499554, 15.6161165, -13.5034256, 13.6009941
28: -1.4954801, 15.5081291, -1.5890355, 15.5123386, -14.1886063, 14.2574387
29: -2.0619454, 12.6842451, -2.1308923, 12.6885271, -11.2747803, 11.3534813
30: -8.1655922, 14.8314104, -8.2933769, 14.8376637, -20.1528168, 20.2799225
31: 0.5096679, 16.0283527, 0.4125843, 16.0343227, -14.3441696, 14.3906670
32: -22.0183792, 2.0217557, -22.0373230, 2.0617361, -18.9851303, 18.9672394
33: -39.7317429, -10.4507799, -39.7610703, -10.3176813, -21.1258011, 21.0176392
34: -33.4297791, -10.0748501, -33.4641457, -10.0108757, -17.5870209, 17.5475998
35: -24.0850983, -0.8070560, -24.1038246, -0.7572064, -18.8298111, 18.8087730
36: -20.8091621, 5.2922292, -20.8310814, 5.3092833, -20.1175613, 20.1572800
37: -32.3139000, -2.6834745, -32.3655090, -2.6709461, -25.9824371, 26.1152115
38: -28.7774010, 0.6368747, -28.8411198, 0.6618280, -24.4466629, 24.5558548
39: -43.9915886, -10.2438221, -44.0193253, -10.1107225, -23.2618179, 23.1620178
40: -31.3526363, -13.0271149, -31.3726406, -13.0135098, -12.9516602, 13.0042686
41: -19.8951206, 2.0573688, -19.9265976, 2.0897784, -18.8185272, 18.8265152
42: -20.1063538, -3.5311997, -20.1457119, -3.5037613, -13.5968170, 13.6672401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=146, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1671

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1693

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5057255, upper bound: 10.5101926
time: 23.06 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5101921, upper bound: 10.5101926
time: 69.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 94.72 seconds
IS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 94.72
Output dim: 18, lower bound: -10.4689578, upper bound: 10.5081133
IS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 94.72
Output dim: 18, lower bound: -10.4733922, upper bound: 10.5081133
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 94.72
Output dim: 18, lower bound: -10.4825028, upper bound: 10.5081133
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 94.72
Output dim: 18, lower bound: -10.4869362, upper bound: 10.5081133
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 94.72
Output dim: 18, lower bound: -10.4922088, upper bound: 10.5101926
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 94.72
Output dim: 18, lower bound: -10.4966684, upper bound: 10.5101926
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 94.72
Output dim: 18, lower bound: -10.5057255, upper bound: 10.5101926
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 94.72
Output dim: 18, lower bound: -10.5101921, upper bound: 10.5101926

## BFS IS instance: IS_B1_B1_B1

### Backsubstitution after applying IS history:
0: -29.3270702, -2.4071484, -29.2753372, -2.4368944, -17.9636383, 18.0546265
1: -13.7359428, 2.6700923, -13.7118721, 2.6427121, -11.7517128, 11.8220215
2: -12.0188751, 4.0657625, -11.9958181, 4.0463462, -10.5003319, 10.5586815
3: -21.1030045, -0.8832216, -21.0584011, -0.9085650, -16.1342964, 16.1449051
4: -19.4284058, 2.6935372, -19.3941135, 2.6735430, -14.3047028, 14.3953896
5: -15.5092316, 4.2210369, -15.4640646, 4.1929550, -15.2430725, 15.2947998
6: -21.6532421, -0.4495554, -21.6238976, -0.4827137, -16.6463089, 16.5602074
7: -18.7223721, 2.4175644, -18.6795063, 2.3885856, -16.8255844, 16.8900681
8: -28.8648396, -1.4109921, -28.8212318, -1.4375701, -17.6746750, 17.7934647
9: -19.0962906, 2.4697819, -19.0399952, 2.4378676, -17.7143021, 17.7571716
10: -16.8532448, 5.3503361, -16.8331070, 5.3218198, -19.7897949, 19.8145905
11: -2.7094307, 15.7621946, -2.6799567, 15.7050867, -17.0330963, 17.0834427
12: -17.3271141, 13.1061783, -17.2791176, 13.0220232, -24.1679306, 24.0752106
13: -30.3475666, -1.5964150, -30.2853165, -1.6406369, -20.8239670, 20.7531395
14: -34.0718575, 0.2606239, -34.0214615, 0.2044330, -29.1931458, 29.1921082
15: -15.3280935, 5.1737609, -15.2919855, 5.1541147, -18.5500793, 18.5452805
16: -15.4742174, 6.2622132, -15.4350491, 6.2229805, -19.0943069, 19.1666145
17: -23.0387802, 1.7457299, -23.0021973, 1.6651614, -23.0827789, 23.0956726
18: 1.8235731, 23.2206345, 1.8499861, 23.1715984, -18.7459717, 18.7523994
19: -0.8191657, 11.5410652, -0.7973022, 11.5253410, -10.9988022, 10.9972572
20: -4.4467077, 9.5692320, -4.4168811, 9.5454149, -13.2839584, 13.2829018
21: -1.3723669, 15.5612373, -1.3310938, 15.5230808, -15.9111328, 15.9155998
22: -3.0683117, 11.4827967, -3.0414078, 11.4680939, -13.5241051, 13.4872627
23: -1.3564539, 15.6231604, -1.3352141, 15.5986700, -13.1962852, 13.2013474
24: -1.8866153, 16.2705059, -1.8621469, 16.2359238, -14.9945145, 15.0093117
25: -2.7005606, 16.3914547, -2.6725636, 16.3677826, -17.4741440, 17.4680061
26: -5.3745537, 21.1228790, -5.3388400, 21.0725517, -25.3270721, 25.3186569
27: -0.4345398, 15.5673437, -0.4012780, 15.5257111, -13.4058075, 13.4195251
28: -1.4769187, 15.4802504, -1.4556508, 15.4573383, -14.1011887, 14.0986481
29: -2.0482895, 12.6637888, -2.0347657, 12.6440105, -11.2110786, 11.1866341
30: -8.1426334, 14.7737951, -8.1154652, 14.7274952, -20.0260162, 20.0799255
31: 0.5352221, 16.0028858, 0.5627255, 15.9830084, -14.2430344, 14.2413330
32: -21.9584007, 2.0084796, -21.9101925, 1.9682345, -18.8348541, 18.7306366
33: -39.6328049, -10.4662962, -39.5468979, -10.5150871, -20.8251038, 20.7001648
34: -33.3481255, -10.0875425, -33.2827606, -10.1347446, -17.3713455, 17.2545013
35: -24.0254536, -0.8176973, -23.9689026, -0.8544872, -18.6678391, 18.5757217
36: -20.7515106, 5.2827482, -20.7032490, 5.2449255, -20.0002823, 19.9212341
37: -32.2323341, -2.6924419, -32.1631279, -2.7387075, -25.8351593, 25.8269653
38: -28.7224197, 0.6223454, -28.6735191, 0.5851631, -24.3059616, 24.2440033
39: -43.8839645, -10.2545547, -43.7943611, -10.2967501, -22.9736252, 22.8763657
40: -31.2946987, -13.0349293, -31.2465916, -13.0658493, -12.8403244, 12.8084984
41: -19.8450508, 2.0430574, -19.8043098, 2.0096421, -18.6857071, 18.6432419
42: -20.0705490, -3.5436080, -20.0422478, -3.5663860, -13.5128746, 13.4703255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=145, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=161, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1594

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B1_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4623863, upper bound: 10.4794969
time: 24.94 seconds

## Relational analysis of IS_B1_B1_B1_B2

### Relational analysis result of IS_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4680702, upper bound: 10.5068191
time: 22.69 seconds

## BFS IS instance: IS_B1_B1_B2

### Backsubstitution after applying IS history:
0: -29.3293571, -2.3809195, -29.2919159, -2.3905120, -18.1149368, 18.1007500
1: -13.7376175, 2.6998894, -13.7271328, 2.6949129, -11.8613358, 11.8698425
2: -12.0198917, 4.0844612, -12.0041370, 4.0790663, -10.5870819, 10.5870399
3: -21.1049957, -0.8650055, -21.0675125, -0.8747697, -16.2066345, 16.1866837
4: -19.4322891, 2.7086940, -19.4027214, 2.7004476, -14.4558029, 14.4367027
5: -15.5109911, 4.2451754, -15.4766340, 4.2367163, -15.3447304, 15.3331757
6: -21.6822014, -0.4466352, -21.6740608, -0.4603977, -16.6988068, 16.6943054
7: -18.7252197, 2.4449887, -18.6995049, 2.4377952, -16.9478302, 16.9377899
8: -28.8678360, -1.3799424, -28.8450432, -1.3846030, -17.8479462, 17.8494148
9: -19.1003208, 2.4965672, -19.0587692, 2.4852588, -17.8364868, 17.8049507
10: -16.8595085, 5.3707390, -16.8448029, 5.3599567, -19.8511124, 19.8493423
11: -2.7183404, 15.7850170, -2.7039409, 15.7457695, -17.0641251, 17.1124115
12: -17.3744736, 13.1101341, -17.3619690, 13.0577602, -24.2532120, 24.2732544
13: -30.3915710, -1.5912528, -30.3630066, -1.6064725, -20.8977890, 20.8389397
14: -34.0822487, 0.2810411, -34.0566635, 0.2426596, -29.2187347, 29.2541962
15: -15.3341007, 5.1835403, -15.3064928, 5.1720419, -18.5874329, 18.5740242
16: -15.4809036, 6.3049726, -15.4611149, 6.2968721, -19.2361069, 19.2376289
17: -23.0708809, 1.7491028, -23.0584869, 1.6944034, -23.1447601, 23.1845627
18: 1.8165197, 23.2397423, 1.8308620, 23.2051411, -18.7572021, 18.7703476
19: -0.8309150, 11.5567951, -0.8215799, 11.5526943, -11.0350800, 11.0333691
20: -4.4579296, 9.5860338, -4.4436579, 9.5754042, -13.3192673, 13.3234406
21: -1.3855252, 15.5938492, -1.3720131, 15.5800381, -15.9603310, 15.9858818
22: -3.0874166, 11.4845543, -3.0774260, 11.4769478, -13.5509796, 13.5418053
23: -1.3657274, 15.6440983, -1.3575149, 15.6351871, -13.2300987, 13.2431183
24: -1.8941727, 16.2899742, -1.8849335, 16.2707043, -15.0257187, 15.0454674
25: -2.7125854, 16.4117565, -2.7004528, 16.4028282, -17.5119858, 17.5169067
26: -5.3911266, 21.1404266, -5.3747997, 21.1036224, -25.3537140, 25.3599014
27: -0.4388490, 15.5880680, -0.4247518, 15.5621967, -13.4266777, 13.4592056
28: -1.4858160, 15.4980927, -1.4764218, 15.4882250, -14.1311150, 14.1421738
29: -2.0560246, 12.6653919, -2.0500708, 12.6472130, -11.2432480, 11.2446632
30: -8.1521912, 14.7937851, -8.1388245, 14.7641449, -20.0734863, 20.0922012
31: 0.5231175, 16.0206871, 0.5358644, 16.0139503, -14.2788773, 14.2787247
32: -22.0032539, 2.0135059, -21.9874992, 2.0039811, -18.9158096, 18.8686447
33: -39.7005310, -10.4635286, -39.6666489, -10.4765711, -20.9403496, 20.8202858
34: -33.4029350, -10.0830498, -33.3793678, -10.0919676, -17.4731598, 17.3880081
35: -24.0723152, -0.8153558, -24.0533772, -0.8245654, -18.7519302, 18.6874466
36: -20.8004761, 5.2846498, -20.7898331, 5.2764874, -20.0830536, 20.0506592
37: -32.2946739, -2.6913242, -32.2750893, -2.7054758, -25.9360352, 25.8945236
38: -28.7623692, 0.6256728, -28.7471237, 0.6116905, -24.3815231, 24.3758469
39: -43.9528351, -10.2525330, -43.9173164, -10.2612419, -23.0878830, 22.9699287
40: -31.3356361, -13.0337811, -31.3191109, -13.0399361, -12.9113197, 12.8538971
41: -19.8815727, 2.0466485, -19.8687935, 2.0360675, -18.7510605, 18.7242432
42: -20.0942917, -3.5394354, -20.0837326, -3.5489101, -13.5475731, 13.5419922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=145, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B1_B1_B2_B1

### Relational analysis result of IS_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4668199, upper bound: 10.4794969
time: 18.47 seconds

## Relational analysis of IS_B1_B1_B2_B2

### Relational analysis result of IS_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4725041, upper bound: 10.5068191
time: 20.68 seconds

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: -29.3414307, -2.4058228, -29.3038998, -2.3659887, -18.0501823, 18.0774231
1: -13.7417488, 2.6710851, -13.7237730, 2.6818385, -11.7973289, 11.8312798
2: -12.0265293, 4.0666070, -12.0107164, 4.0931215, -10.5550346, 10.5716591
3: -21.1159515, -0.8817158, -21.0842628, -0.8352976, -16.2224960, 16.1664963
4: -19.4424038, 2.6946001, -19.4233284, 2.7332387, -14.3768196, 14.4180107
5: -15.5203600, 4.2221899, -15.4865799, 4.2588263, -15.3212433, 15.3138428
6: -21.6544456, -0.4466467, -21.6486359, -0.4722185, -16.6568680, 16.5919724
7: -18.7340126, 2.4182773, -18.7026234, 2.4348984, -16.8838577, 16.9093094
8: -28.8774357, -1.4099507, -28.8478470, -1.3866606, -17.7384720, 17.8147888
9: -19.1133671, 2.4709110, -19.0765476, 2.5108111, -17.8043594, 17.7884521
10: -16.8552742, 5.3515954, -16.8738289, 5.3518262, -19.8184662, 19.8625755
11: -2.7111051, 15.7774429, -2.7500165, 15.7379532, -17.0592995, 17.1641960
12: -17.3293037, 13.1187153, -17.3539543, 13.0515108, -24.1989975, 24.1629257
13: -30.3576756, -1.5942330, -30.3089619, -1.5793171, -20.8968582, 20.7724724
14: -34.0771217, 0.2615285, -34.0625114, 0.2255163, -29.2188568, 29.2436829
15: -15.3332720, 5.1758695, -15.3076582, 5.1998620, -18.6005936, 18.5628395
16: -15.4845276, 6.2628736, -15.4652405, 6.2814083, -19.1628227, 19.1934738
17: -23.0406246, 1.7483580, -23.0424461, 1.6874106, -23.1056442, 23.1407013
18: 1.8209033, 23.2295246, 1.7655764, 23.1975098, -18.7697754, 18.8462677
19: -0.8206053, 11.5413036, -0.8378916, 11.5302391, -11.0104637, 11.0357265
20: -4.4495311, 9.5694542, -4.4621572, 9.5502682, -13.2966766, 13.3271790
21: -1.3748498, 15.5618229, -1.3671966, 15.5308714, -15.9268188, 15.9508171
22: -3.0695477, 11.4841480, -3.0695746, 11.4714603, -13.5355682, 13.5175743
23: -1.3579912, 15.6315937, -1.3892531, 15.6160307, -13.2082825, 13.2589073
24: -1.8875446, 16.2791977, -1.9206085, 16.2586575, -15.0130539, 15.0742073
25: -2.7021747, 16.3934250, -2.7235966, 16.3761482, -17.4850998, 17.5169907
26: -5.3767323, 21.1247292, -5.4315605, 21.0866222, -25.3449097, 25.4174728
27: -0.4362431, 15.5732565, -0.4496422, 15.5404148, -13.4181595, 13.4716759
28: -1.4787211, 15.4829016, -1.5006709, 15.4664774, -14.1148987, 14.1431389
29: -2.0488343, 12.6719131, -2.0676403, 12.6621628, -11.2251892, 11.2290993
30: -8.1444073, 14.7822771, -8.1637230, 14.7470388, -20.0441589, 20.1331177
31: 0.5329633, 16.0039444, 0.5123148, 15.9882412, -14.2579308, 14.2915573
32: -21.9583454, 2.0090723, -21.9276886, 1.9716339, -18.8397217, 18.7542343
33: -39.6292953, -10.4644928, -39.5647392, -10.4689760, -20.8657608, 20.7281189
34: -33.3495407, -10.0861149, -33.3155899, -10.1266289, -17.3836288, 17.2942619
35: -24.0234814, -0.8158123, -23.9806156, -0.8282657, -18.6927795, 18.5953064
36: -20.7527790, 5.2831354, -20.7209492, 5.2491770, -20.0050354, 19.9450684
37: -32.2346992, -2.6964145, -32.1999512, -2.7405262, -25.8425140, 25.8769531
38: -28.7257137, 0.6245837, -28.7272491, 0.5962691, -24.3199921, 24.3183899
39: -43.8912811, -10.2537632, -43.8293037, -10.2548227, -23.0159607, 22.9122620
40: -31.2969284, -13.0346603, -31.2671700, -13.0621872, -12.8462410, 12.8356514
41: -19.8461800, 2.0463240, -19.8345127, 2.0219538, -18.6998825, 18.6810150
42: -20.0717735, -3.5422223, -20.0744057, -3.5574534, -13.5229492, 13.5117149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=145, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=161, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1594

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B1_B2_B1_B1

### Relational analysis result of IS_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4756205, upper bound: 10.4794969
time: 26.71 seconds

## Relational analysis of IS_B1_B2_B1_B2

### Relational analysis result of IS_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4812070, upper bound: 10.5068191
time: 22.74 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -29.3437004, -2.3796139, -29.3204346, -2.3196020, -18.2014923, 18.1235428
1: -13.7434521, 2.7008770, -13.7390537, 2.7340460, -11.9069672, 11.8790932
2: -12.0275526, 4.0852861, -12.0190258, 4.1258364, -10.6417999, 10.6000214
3: -21.1179714, -0.8634710, -21.0933781, -0.8014774, -16.2948418, 16.2082672
4: -19.4462662, 2.7097974, -19.4319572, 2.7601128, -14.5279198, 14.4593239
5: -15.5221090, 4.2462978, -15.4991550, 4.3025551, -15.4228592, 15.3522377
6: -21.6834126, -0.4437494, -21.6988316, -0.4498558, -16.7093353, 16.7260590
7: -18.7368431, 2.4456987, -18.7226181, 2.4841075, -17.0060577, 16.9570389
8: -28.8804092, -1.3789654, -28.8716393, -1.3336916, -17.9117584, 17.8707657
9: -19.1173706, 2.4977262, -19.0952778, 2.5581412, -17.9265442, 17.8362503
10: -16.8615513, 5.3719759, -16.8855267, 5.3899612, -19.8798065, 19.8973503
11: -2.7200181, 15.8002567, -2.7739739, 15.7785978, -17.0902519, 17.1932373
12: -17.3766918, 13.1226501, -17.4367828, 13.0872765, -24.2842712, 24.3609390
13: -30.4017296, -1.5891018, -30.3866882, -1.5451632, -20.9706268, 20.8582497
14: -34.0875015, 0.2819295, -34.0977631, 0.2637367, -29.2444077, 29.3057938
15: -15.3392620, 5.1856370, -15.3221989, 5.2177958, -18.6380539, 18.5915756
16: -15.4911938, 6.3056455, -15.4912739, 6.3553238, -19.3046417, 19.2644615
17: -23.0727119, 1.7517281, -23.0987720, 1.7166760, -23.1676407, 23.2296677
18: 1.8138814, 23.2486267, 1.7464805, 23.2310562, -18.7810478, 18.8642311
19: -0.8323603, 11.5570354, -0.8621655, 11.5575991, -11.0467415, 11.0718346
20: -4.4607687, 9.5862560, -4.4888935, 9.5802689, -13.3319778, 13.3677444
21: -1.3880038, 15.5944643, -1.4081454, 15.5877800, -15.9760399, 16.0211105
22: -3.0886533, 11.4858923, -3.1056051, 11.4803343, -13.5624313, 13.5721207
23: -1.3673000, 15.6525402, -1.4115276, 15.6525469, -13.2420921, 13.3006687
24: -1.8951020, 16.2986870, -1.9434452, 16.2934361, -15.0442696, 15.1104012
25: -2.7142406, 16.4137440, -2.7514620, 16.4112015, -17.5229645, 17.5659180
26: -5.3933039, 21.1422958, -5.4675326, 21.1176891, -25.3716049, 25.4587402
27: -0.4405618, 15.5939941, -0.4731131, 15.5769196, -13.4390602, 13.5114021
28: -1.4876151, 15.5007496, -1.5214748, 15.4973850, -14.1448250, 14.1866989
29: -2.0565689, 12.6734686, -2.0829787, 12.6653557, -11.2573776, 11.2871056
30: -8.1539660, 14.8022537, -8.1870823, 14.7837191, -20.0915985, 20.1454773
31: 0.5208735, 16.0217533, 0.4854574, 16.0191650, -14.2937546, 14.3289833
32: -22.0031853, 2.0140643, -22.0050220, 2.0074039, -18.9206924, 18.8922653
33: -39.6969681, -10.4617386, -39.6844559, -10.4304752, -20.9810371, 20.8482361
34: -33.4043655, -10.0816040, -33.4122620, -10.0838318, -17.4854507, 17.4277992
35: -24.0703793, -0.8134551, -24.0650864, -0.7983401, -18.7769089, 18.7069893
36: -20.8017197, 5.2850275, -20.8075104, 5.2807083, -20.0878601, 20.0745010
37: -32.2970200, -2.6952543, -32.3119431, -2.7072926, -25.9433746, 25.9445343
38: -28.7655602, 0.6279459, -28.8008366, 0.6228118, -24.3955765, 24.4501724
39: -43.9601822, -10.2517433, -43.9522171, -10.2192898, -23.1302414, 23.0058365
40: -31.3378944, -13.0335350, -31.3396893, -13.0362320, -12.9172325, 12.8810425
41: -19.8826790, 2.0499270, -19.8990440, 2.0483751, -18.7652512, 18.7620087
42: -20.0955238, -3.5380731, -20.1159325, -3.5399709, -13.5576439, 13.5834160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=145, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B1_B2_B2_B1

### Relational analysis result of IS_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4800480, upper bound: 10.4794969
time: 23.45 seconds

## Relational analysis of IS_B1_B2_B2_B2

### Relational analysis result of IS_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4856412, upper bound: 10.5068191
time: 23.78 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -29.3555584, -2.3992157, -29.3321972, -2.3513846, -18.0856781, 18.1199532
1: -13.7403431, 2.6728697, -13.7228241, 2.6720948, -11.7871933, 11.8373718
2: -12.0284538, 4.0695314, -12.0167809, 4.0872989, -10.5548897, 10.5836964
3: -21.1338158, -0.8766718, -21.1163445, -0.8176169, -16.2578697, 16.1996498
4: -19.4486408, 2.6998739, -19.4342308, 2.7370362, -14.3941231, 14.4412613
5: -15.5374355, 4.2277918, -15.5194168, 4.2724352, -15.3521347, 15.3497467
6: -21.6595020, -0.4446726, -21.6557159, -0.4581642, -16.6838455, 16.6331863
7: -18.7388268, 2.4231534, -18.7164478, 2.4190836, -16.8744659, 16.9299698
8: -28.8764057, -1.4086900, -28.8484230, -1.4245529, -17.7018013, 17.8227463
9: -19.1279716, 2.4777594, -19.1039066, 2.5365913, -17.8421021, 17.8250275
10: -16.8656006, 5.3580251, -16.8748035, 5.3822727, -19.8677902, 19.8655586
11: -2.7223725, 15.7957153, -2.8151207, 15.7661552, -17.0988007, 17.2443390
12: -17.3369541, 13.1541901, -17.3856659, 13.1235142, -24.2758408, 24.2294312
13: -30.3603325, -1.5842490, -30.3178749, -1.5514269, -20.9192963, 20.7962265
14: -34.0922050, 0.3070607, -34.1909866, 0.2940907, -29.2893219, 29.4101562
15: -15.3511963, 5.1829448, -15.3411112, 5.2327561, -18.6392517, 18.5932541
16: -15.4846039, 6.2673306, -15.4759541, 6.2620478, -19.1532364, 19.2204781
17: -23.0472813, 1.8073130, -23.1369209, 1.7809432, -23.2039108, 23.2944183
18: 1.8149190, 23.2518349, 1.7437520, 23.2336311, -18.8031006, 18.8816147
19: -0.8265600, 11.5440464, -0.8617096, 11.5323448, -11.0347614, 11.0578518
20: -4.4582028, 9.5802879, -4.5076308, 9.5691204, -13.3129158, 13.3806782
21: -1.3836856, 15.5769463, -1.4333925, 15.5534382, -15.9418259, 16.0278740
22: -3.0756600, 11.4894381, -3.0720804, 11.4858131, -13.5522461, 13.5242805
23: -1.3621941, 15.6247597, -1.3947058, 15.6028013, -13.2162666, 13.2644310
24: -1.8925500, 16.2811012, -1.9250517, 16.2580833, -15.0179062, 15.0797424
25: -2.7094517, 16.3966732, -2.7247667, 16.3780479, -17.4939270, 17.5264320
26: -5.3858142, 21.1583328, -5.4928436, 21.1447201, -25.4021835, 25.5094528
27: -0.4454474, 15.5852671, -0.4774418, 15.5635900, -13.4481392, 13.5076141
28: -1.4842358, 15.4869585, -1.5222406, 15.4709997, -14.1324959, 14.1677856
29: -2.0531688, 12.6744299, -2.0817034, 12.6669636, -11.2366867, 11.2495346
30: -8.1537504, 14.8022842, -8.2207546, 14.7801228, -20.0849915, 20.2127075
31: 0.5247192, 16.0089073, 0.4911962, 15.9970388, -14.2850571, 14.3009872
32: -21.9721622, 2.0158501, -21.9397507, 2.0220714, -18.8972778, 18.7604675
33: -39.6656647, -10.4556189, -39.6196518, -10.4029722, -20.9686661, 20.7708588
34: -33.3717957, -10.0811119, -33.3312378, -10.0624313, -17.4713135, 17.2973213
35: -24.0387421, -0.8116403, -24.0048256, -0.8138537, -18.7198334, 18.6187935
36: -20.7579689, 5.2898097, -20.7250824, 5.2731352, -20.0291061, 19.9658432
37: -32.2479630, -2.6808758, -32.2143860, -2.7027311, -25.8727722, 25.9422913
38: -28.7329063, 0.6309991, -28.7115517, 0.6236176, -24.3554077, 24.3308258
39: -43.9134445, -10.2468910, -43.8575478, -10.1887836, -23.1041565, 22.9347305
40: -31.3080635, -13.0286865, -31.2768288, -13.0434437, -12.8737564, 12.8725662
41: -19.8563690, 2.0502830, -19.8296909, 2.0505381, -18.7372971, 18.6796722
42: -20.0805931, -3.5371015, -20.0704689, -3.5308483, -13.5504990, 13.5426178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=145, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=161, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1594

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4855481, upper bound: 10.4815068
time: 24.42 seconds

## Relational analysis of IS_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4913295, upper bound: 10.5089087
time: 23.20 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -29.3577957, -2.3729901, -29.3487625, -2.3049836, -18.2369995, 18.1660690
1: -13.7420273, 2.7026722, -13.7381153, 2.7242723, -11.8968163, 11.8851929
2: -12.0294809, 4.0882530, -12.0251303, 4.1200304, -10.6416473, 10.6120663
3: -21.1357803, -0.8584676, -21.1254425, -0.7837901, -16.3302383, 16.2414322
4: -19.4524803, 2.7150054, -19.4428635, 2.7639217, -14.5452347, 14.4826050
5: -15.5391922, 4.2519403, -15.5320148, 4.3161602, -15.4537621, 15.3881378
6: -21.6884422, -0.4417820, -21.7059116, -0.4358501, -16.7363739, 16.7673111
7: -18.7416630, 2.4505777, -18.7364349, 2.4682927, -16.9966583, 16.9777222
8: -28.8793945, -1.3777041, -28.8722382, -1.3716183, -17.8751144, 17.8787155
9: -19.1319447, 2.5045230, -19.1226578, 2.5839090, -17.9643021, 17.8728104
10: -16.8718796, 5.3784208, -16.8864861, 5.4204116, -19.9291573, 19.9002914
11: -2.7313266, 15.8185329, -2.8390839, 15.8068428, -17.1297455, 17.2734299
12: -17.3842258, 13.1581335, -17.4684544, 13.1592703, -24.3611755, 24.4274597
13: -30.4043751, -1.5791011, -30.3955555, -1.5172629, -20.9930344, 20.8819923
14: -34.1025505, 0.3274565, -34.2262650, 0.3322926, -29.3148956, 29.4722366
15: -15.3572083, 5.1927342, -15.3556461, 5.2507515, -18.6766663, 18.6220055
16: -15.4913177, 6.3101106, -15.5020008, 6.3359509, -19.2950325, 19.2914772
17: -23.0793247, 1.8106282, -23.1931858, 1.8102064, -23.2658997, 23.3833313
18: 1.8078938, 23.2709579, 1.7245626, 23.2671700, -18.8142471, 18.8995628
19: -0.8383064, 11.5597744, -0.8859906, 11.5596962, -11.0709934, 11.0939636
20: -4.4694409, 9.5970659, -4.5343685, 9.5991096, -13.3481789, 13.4212112
21: -1.3968229, 15.6095695, -1.4743023, 15.6103659, -15.9910774, 16.0981750
22: -3.0947585, 11.4911804, -3.1081047, 11.4947109, -13.5790939, 13.5788155
23: -1.3715000, 15.6456938, -1.4170589, 15.6393557, -13.2500534, 13.3061752
24: -1.9000783, 16.3005733, -1.9478760, 16.2928791, -15.0491028, 15.1158905
25: -2.7215161, 16.4169865, -2.7526827, 16.4131031, -17.5317612, 17.5753517
26: -5.4024301, 21.1758499, -5.5288181, 21.1757889, -25.4287415, 25.5507507
27: -0.4497647, 15.6060209, -0.5009055, 15.6000767, -13.4690094, 13.5473175
28: -1.4931560, 15.5048084, -1.5430226, 15.5019188, -14.1624489, 14.2113457
29: -2.0609059, 12.6760416, -2.0970149, 12.6701756, -11.2688637, 11.3075676
30: -8.1633186, 14.8222857, -8.2441502, 14.8167686, -20.1324463, 20.2249985
31: 0.5126266, 16.0267220, 0.4642916, 16.0279694, -14.3208771, 14.3384361
32: -22.0170059, 2.0208917, -22.0170994, 2.0578585, -18.9782867, 18.8985214
33: -39.7332687, -10.4528761, -39.7393494, -10.3644562, -21.0839386, 20.8909073
34: -33.4266510, -10.0765972, -33.4278679, -10.0196028, -17.5731468, 17.4308319
35: -24.0855923, -0.8092523, -24.0892792, -0.7839887, -18.8038788, 18.7304764
36: -20.8069572, 5.2916636, -20.8115730, 5.3046684, -20.1118546, 20.0952377
37: -32.3102913, -2.6797299, -32.3263206, -2.6695027, -25.9736633, 26.0098572
38: -28.7728310, 0.6343427, -28.7850533, 0.6502018, -24.4310226, 24.4625320
39: -43.9822311, -10.2448730, -43.9804688, -10.1532574, -23.2183609, 23.0282440
40: -31.3489838, -13.0275259, -31.3493671, -13.0174599, -12.9447174, 12.9179611
41: -19.8928795, 2.0538614, -19.8941593, 2.0769765, -18.8026886, 18.7606583
42: -20.1043339, -3.5329247, -20.1119881, -3.5133414, -13.5851898, 13.6143188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=145, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1399

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B2_B1_B2_B1

### Relational analysis result of IS_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4900086, upper bound: 10.4815068
time: 23.69 seconds

## Relational analysis of IS_B2_B1_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4957871, upper bound: 10.5089087
time: 20.76 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: -29.3698692, -2.3979063, -29.3606892, -2.2804551, -18.1722031, 18.1427269
1: -13.7461901, 2.6738594, -13.7347517, 2.7112129, -11.8328247, 11.8466187
2: -12.0361176, 4.0704126, -12.0316868, 4.1340628, -10.6095924, 10.5966778
3: -21.1467533, -0.8751526, -21.1421909, -0.7443085, -16.3460922, 16.2212257
4: -19.4626007, 2.7008934, -19.4634590, 2.7967033, -14.4662285, 14.4638939
5: -15.5485992, 4.2289362, -15.5419502, 4.3382912, -15.4303017, 15.3688126
6: -21.6607170, -0.4417324, -21.6804295, -0.4476566, -16.6944199, 16.6649323
7: -18.7504425, 2.4238424, -18.7395782, 2.4654379, -16.9327240, 16.9492416
8: -28.8890419, -1.4076700, -28.8750896, -1.3736553, -17.7656403, 17.8440552
9: -19.1450100, 2.4788740, -19.1404572, 2.6095028, -17.9321671, 17.8562775
10: -16.8676147, 5.3592496, -16.9154968, 5.4122624, -19.8964767, 19.9134712
11: -2.7240248, 15.8109531, -2.8851144, 15.7990379, -17.1249619, 17.3250809
12: -17.3390770, 13.1666908, -17.4604874, 13.1530104, -24.3069153, 24.3170853
13: -30.3704739, -1.5820799, -30.3415146, -1.4901743, -20.9921341, 20.8155518
14: -34.0974731, 0.3079872, -34.2320328, 0.3151674, -29.3149567, 29.4617996
15: -15.3563576, 5.1850414, -15.3568497, 5.2785349, -18.6897888, 18.6108665
16: -15.4949045, 6.2679987, -15.5061817, 6.3204517, -19.2217751, 19.2473602
17: -23.0491104, 1.8099470, -23.1771908, 1.8032084, -23.2268372, 23.3395462
18: 1.8122902, 23.2607536, 1.6593719, 23.2595119, -18.8269043, 18.9754982
19: -0.8280096, 11.5442848, -0.9023046, 11.5372486, -11.0464439, 11.0963287
20: -4.4610243, 9.5804977, -4.5529060, 9.5739908, -13.3256378, 13.4250069
21: -1.3861589, 15.5775547, -1.4695358, 15.5612040, -15.9575043, 16.0631371
22: -3.0768931, 11.4907417, -3.1002409, 11.4892578, -13.5636902, 13.5546074
23: -1.3637295, 15.6331921, -1.4487815, 15.6202002, -13.2282715, 13.3220196
24: -1.8934669, 16.2897949, -1.9835486, 16.2808228, -15.0364532, 15.1446381
25: -2.7110901, 16.3986683, -2.7758574, 16.3864441, -17.5049362, 17.5754242
26: -5.3880420, 21.1601791, -5.5855737, 21.1587887, -25.4200211, 25.6083679
27: -0.4471502, 15.5911951, -0.5257306, 15.5782986, -13.4604874, 13.5597649
28: -1.4860353, 15.4896212, -1.5672774, 15.4801674, -14.1462250, 14.2123070
29: -2.0536819, 12.6825409, -2.1146045, 12.6851521, -11.2508202, 11.2920036
30: -8.1555347, 14.8107595, -8.2690439, 14.7996979, -20.1031494, 20.2658920
31: 0.5224710, 16.0099697, 0.4407582, 16.0022774, -14.2999382, 14.3512497
32: -21.9721336, 2.0164485, -21.9572201, 2.0254855, -18.9021530, 18.7840424
33: -39.6621323, -10.4538651, -39.6374969, -10.3568954, -21.0093651, 20.7987823
34: -33.3732529, -10.0796566, -33.3641167, -10.0542393, -17.4836464, 17.3370895
35: -24.0367661, -0.8097026, -24.0165291, -0.7876267, -18.7448120, 18.6383972
36: -20.7592468, 5.2901530, -20.7427425, 5.2773561, -20.0337982, 19.9897079
37: -32.2503510, -2.6848569, -32.2512703, -2.7046137, -25.8800964, 25.9922867
38: -28.7361832, 0.6332345, -28.7652397, 0.6347389, -24.3695068, 24.4051514
39: -43.9207535, -10.2461367, -43.8924866, -10.1467466, -23.1465225, 22.9706345
40: -31.3102779, -13.0284090, -31.2974472, -13.0397778, -12.8796692, 12.8996887
41: -19.8574524, 2.0535669, -19.8598671, 2.0628800, -18.7514648, 18.7173920
42: -20.0818157, -3.5357094, -20.1026287, -3.5218778, -13.5605888, 13.5839920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=145, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=161, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1671

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B2_B2_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4987546, upper bound: 10.4815068
time: 44.74 seconds

## Relational analysis of IS_B2_B2_B1_B2

### Relational analysis result of IS_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5044440, upper bound: 10.5089087
time: 24.65 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: -29.3721542, -2.3716698, -29.3772736, -2.2340961, -18.3234863, 18.1888618
1: -13.7478561, 2.7036684, -13.7500343, 2.7634134, -11.9424515, 11.8944588
2: -12.0371590, 4.0891190, -12.0399914, 4.1668005, -10.6963806, 10.6250458
3: -21.1487465, -0.8569245, -21.1512909, -0.7105079, -16.4184570, 16.2630386
4: -19.4664650, 2.7160850, -19.4721222, 2.8235745, -14.6173401, 14.5052071
5: -15.5503302, 4.2530956, -15.5544910, 4.3820210, -15.5319061, 15.4071732
6: -21.6896954, -0.4388418, -21.7306232, -0.4253359, -16.7469559, 16.7990303
7: -18.7532997, 2.4512863, -18.7595596, 2.5145903, -17.0549011, 16.9969788
8: -28.8920212, -1.3767152, -28.8989182, -1.3207221, -17.9389420, 17.9000969
9: -19.1490211, 2.5056453, -19.1591625, 2.6568053, -18.0543518, 17.9040756
10: -16.8739090, 5.3796463, -16.9271660, 5.4504237, -19.9578819, 19.9482346
11: -2.7329893, 15.8337708, -2.9090481, 15.8397026, -17.1559143, 17.3542366
12: -17.3864098, 13.1706944, -17.5433159, 13.1887903, -24.3922195, 24.5151443
13: -30.4145622, -1.5769143, -30.4192734, -1.4559708, -21.0659027, 20.9013481
14: -34.1078033, 0.3283844, -34.2673340, 0.3533516, -29.3405151, 29.5238647
15: -15.3623495, 5.1948237, -15.3714113, 5.2965117, -18.7272873, 18.6396027
16: -15.5016155, 6.3107457, -15.5321884, 6.3943729, -19.3635826, 19.3183289
17: -23.0811577, 1.8133371, -23.2334633, 1.8324935, -23.2888184, 23.4284439
18: 1.8052382, 23.2798653, 1.6401458, 23.2930717, -18.8380890, 18.9934235
19: -0.8397560, 11.5600262, -0.9265957, 11.5645924, -11.0826759, 11.1324577
20: -4.4722624, 9.5972891, -4.5796404, 9.6039867, -13.3608856, 13.4655571
21: -1.3993468, 15.6102047, -1.5104637, 15.6181288, -16.0067749, 16.1334267
22: -3.0959790, 11.4925175, -3.1362536, 11.4981213, -13.5905800, 13.6091499
23: -1.3730459, 15.6541252, -1.4710855, 15.6567078, -13.2620583, 13.3637772
24: -1.9010210, 16.3092709, -2.0063701, 16.3155823, -15.0676308, 15.1808014
25: -2.7231774, 16.4189587, -2.8037553, 16.4215164, -17.5427628, 17.6243439
26: -5.4046445, 21.1776848, -5.6215410, 21.1898193, -25.4466553, 25.6496353
27: -0.4514685, 15.6119461, -0.5491824, 15.6147652, -13.4813576, 13.5995026
28: -1.4949489, 15.5074930, -1.5880904, 15.5110893, -14.1761665, 14.2558861
29: -2.0614400, 12.6841288, -2.1299198, 12.6883545, -11.2830009, 11.3500061
30: -8.1651039, 14.8307152, -8.2924004, 14.8363266, -20.1505737, 20.2782669
31: 0.5103378, 16.0277748, 0.4138536, 16.0332088, -14.3357582, 14.3886909
32: -22.0169773, 2.0215025, -22.0345783, 2.0612788, -18.9831696, 18.9221497
33: -39.7297516, -10.4511175, -39.7571259, -10.3183470, -21.1246185, 20.9188499
34: -33.4281006, -10.0751467, -33.4607620, -10.0114517, -17.5854530, 17.4706001
35: -24.0836601, -0.8073494, -24.1009598, -0.7577224, -18.8288574, 18.7500305
36: -20.8082199, 5.2920303, -20.8293037, 5.3089013, -20.1166077, 20.1190948
37: -32.3126526, -2.6836700, -32.3631439, -2.6714153, -25.9809723, 26.0598526
38: -28.7760468, 0.6366048, -28.8387718, 0.6612682, -24.4450760, 24.5368958
39: -43.9895706, -10.2440987, -44.0153580, -10.1113510, -23.2607193, 23.0641289
40: -31.3512344, -13.0272627, -31.3699493, -13.0138378, -12.9506454, 12.9450684
41: -19.8940125, 2.0571434, -19.9243698, 2.0893013, -18.8168869, 18.7983932
42: -20.1055279, -3.5315185, -20.1441422, -3.5044172, -13.5952797, 13.6557465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=145, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1671

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_B2_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5032184, upper bound: 10.4815068
time: 23.50 seconds

## Relational analysis of IS_B2_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5089084, upper bound: 10.5089087
time: 25.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 51.02 seconds
IS_B1_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4623863, upper bound: 10.4794969
IS_B1_B1_B1_B2, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4680702, upper bound: 10.5068191
IS_B1_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4668199, upper bound: 10.4794969
IS_B1_B1_B2_B2, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4725041, upper bound: 10.5068191
IS_B1_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4756205, upper bound: 10.4794969
IS_B1_B2_B1_B2, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4812070, upper bound: 10.5068191
IS_B1_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4800480, upper bound: 10.4794969
IS_B1_B2_B2_B2, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4856412, upper bound: 10.5068191
IS_B2_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4855481, upper bound: 10.4815068
IS_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4913295, upper bound: 10.5089087
IS_B2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4900086, upper bound: 10.4815068
IS_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4957871, upper bound: 10.5089087
IS_B2_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.4987546, upper bound: 10.4815068
IS_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.5044440, upper bound: 10.5089087
IS_B2_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.5032184, upper bound: 10.4815068
IS_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 51.02
Output dim: 18, lower bound: -10.5089084, upper bound: 10.5089087

## BFS IS instance: IS_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -29.3553047, -2.4009256, -29.3317680, -2.3548651, -18.0592880, 18.1175537
1: -13.7401543, 2.6717088, -13.7224445, 2.6697495, -11.7738914, 11.8357468
2: -12.0282574, 4.0684929, -12.0164175, 4.0852051, -10.5439949, 10.5822372
3: -21.1334686, -0.8780608, -21.1156693, -0.8204141, -16.2468414, 16.1975250
4: -19.4483833, 2.6984200, -19.4337196, 2.7341743, -14.3730316, 14.4390678
5: -15.5370770, 4.2264743, -15.5187101, 4.2697220, -15.3392296, 15.3476715
6: -21.6584835, -0.4457808, -21.6536503, -0.4602585, -16.6815720, 16.6295395
7: -18.7384739, 2.4221468, -18.7157516, 2.4170485, -16.8589096, 16.9281464
8: -28.8759232, -1.4100976, -28.8473625, -1.4275093, -17.6901016, 17.8197479
9: -19.1276836, 2.4759710, -19.1033630, 2.5329764, -17.8184204, 17.8224754
10: -16.8643608, 5.3572197, -16.8725128, 5.3807435, -19.8637047, 19.8571472
11: -2.7206328, 15.7954416, -2.8115864, 15.7656069, -17.0962601, 17.2265778
12: -17.3353176, 13.1535406, -17.3822441, 13.1222725, -24.2719879, 24.2207489
13: -30.3599396, -1.5871453, -30.3171387, -1.5573797, -20.8899918, 20.7925186
14: -34.0903473, 0.3056583, -34.1872635, 0.2911654, -29.2837219, 29.4029388
15: -15.3507195, 5.1815109, -15.3401852, 5.2299700, -18.6340332, 18.5907288
16: -15.4838390, 6.2658405, -15.4744377, 6.2590523, -19.1374397, 19.2173462
17: -23.0460014, 1.8066421, -23.1343689, 1.7796516, -23.2009277, 23.2837601
18: 1.8170223, 23.2514820, 1.7480512, 23.2328854, -18.8000488, 18.8675842
19: -0.8252444, 11.5437441, -0.8591142, 11.5316982, -11.0319920, 11.0532398
20: -4.4565506, 9.5801640, -4.5043411, 9.5688791, -13.3112984, 13.3770771
21: -1.3821530, 15.5768299, -1.4303579, 15.5531731, -15.9397430, 16.0223999
22: -3.0744643, 11.4891405, -3.0696671, 11.4852390, -13.5503235, 13.5156403
23: -1.3604937, 15.6245489, -1.3913412, 15.6023903, -13.2139702, 13.2481270
24: -1.8908334, 16.2807961, -1.9215231, 16.2574921, -15.0154343, 15.0629730
25: -2.7078605, 16.3964596, -2.7216520, 16.3775978, -17.4915085, 17.5177879
26: -5.3833981, 21.1579170, -5.4879451, 21.1438980, -25.3985138, 25.4971542
27: -0.4438095, 15.5850725, -0.4740653, 15.5632257, -13.4460411, 13.5015564
28: -1.4826279, 15.4868031, -1.5190415, 15.4706459, -14.1303596, 14.1517525
29: -2.0521181, 12.6742563, -2.0795748, 12.6665964, -11.2350006, 11.2303696
30: -8.1517391, 14.8021069, -8.2167397, 14.7797270, -20.0824203, 20.1903687
31: 0.5263219, 16.0087605, 0.4944253, 15.9967527, -14.2829018, 14.2935562
32: -21.9714336, 2.0147362, -21.9381924, 2.0197220, -18.8910980, 18.7566795
33: -39.6648102, -10.4566450, -39.6180496, -10.4049940, -20.9515686, 20.7673721
34: -33.3705711, -10.0815163, -33.3286819, -10.0631924, -17.4691010, 17.2818184
35: -24.0381126, -0.8121998, -24.0036011, -0.8149877, -18.7172623, 18.6182289
36: -20.7574463, 5.2885399, -20.7239246, 5.2707343, -20.0154877, 19.9633636
37: -32.2469940, -2.6822257, -32.2124825, -2.7054467, -25.8596649, 25.9380493
38: -28.7320766, 0.6301064, -28.7097950, 0.6218352, -24.3440857, 24.3270569
39: -43.9125748, -10.2491474, -43.8558197, -10.1933241, -23.0667419, 22.9304123
40: -31.3074360, -13.0291767, -31.2755451, -13.0443554, -12.8576775, 12.8704605
41: -19.8558922, 2.0493569, -19.8287163, 2.0489702, -18.7331848, 18.6769104
42: -20.0796700, -3.5375161, -20.0687256, -3.5316830, -13.5526886, 13.5394096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=144, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=161, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1594

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_B2_B1_B1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4700052, upper bound: 10.5038322
time: 24.41 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4906896, upper bound: 10.5082401
time: 19.28 seconds

## BFS IS instance: IS_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -29.3576050, -2.3747187, -29.3483486, -2.3085184, -18.2106094, 18.1636734
1: -13.7418613, 2.7015107, -13.7377138, 2.7219436, -11.8835144, 11.8835449
2: -12.0292912, 4.0872064, -12.0247345, 4.1179323, -10.6307678, 10.6105995
3: -21.1354580, -0.8598375, -21.1247826, -0.7865891, -16.3191872, 16.2393074
4: -19.4522648, 2.7135973, -19.4423561, 2.7610669, -14.5241356, 14.4803963
5: -15.5388517, 4.2505898, -15.5312548, 4.3134346, -15.4408646, 15.3860550
6: -21.6874733, -0.4428673, -21.7038441, -0.4379520, -16.7341232, 16.7636375
7: -18.7413082, 2.4495769, -18.7357445, 2.4662547, -16.9811401, 16.9758911
8: -28.8789005, -1.3791275, -28.8711929, -1.3745885, -17.8633881, 17.8757286
9: -19.1316929, 2.5027113, -19.1221123, 2.5802791, -17.9405670, 17.8702545
10: -16.8706284, 5.3776135, -16.8842010, 5.4188862, -19.9250565, 19.8918610
11: -2.7295713, 15.8182497, -2.8355439, 15.8062820, -17.1271973, 17.2556419
12: -17.3826294, 13.1575298, -17.4650631, 13.1580162, -24.3572388, 24.4187927
13: -30.4039974, -1.5820336, -30.3948212, -1.5232096, -20.9637451, 20.8782806
14: -34.1007080, 0.3260660, -34.2225800, 0.3293991, -29.3093567, 29.4650955
15: -15.3567324, 5.1912985, -15.3547010, 5.2479448, -18.6714325, 18.6194763
16: -15.4905252, 6.3086200, -15.5004902, 6.3329425, -19.2792282, 19.2883644
17: -23.0780697, 1.8099980, -23.1906605, 1.8089101, -23.2629318, 23.3726959
18: 1.8099852, 23.2705708, 1.7288718, 23.2664242, -18.8111572, 18.8855324
19: -0.8369718, 11.5594721, -0.8833971, 11.5590620, -11.0682220, 11.0893612
20: -4.4677925, 9.5969543, -4.5310764, 9.5988741, -13.3465614, 13.4176369
21: -1.3952971, 15.6094522, -1.4712834, 15.6101160, -15.9889755, 16.0926819
22: -3.0935292, 11.4909067, -3.1056864, 11.4941492, -13.5771828, 13.5701675
23: -1.3698025, 15.6454735, -1.4136696, 15.6389151, -13.2477722, 13.2898788
24: -1.8983712, 16.3002605, -1.9443803, 16.2922630, -15.0466080, 15.0991249
25: -2.7199373, 16.4167633, -2.7495756, 16.4126720, -17.5293732, 17.5666809
26: -5.4000397, 21.1754131, -5.5239401, 21.1749573, -25.4251022, 25.5383835
27: -0.4480863, 15.6058559, -0.4974947, 15.5997124, -13.4668884, 13.5412369
28: -1.4915581, 15.5046453, -1.5398297, 15.5015802, -14.1603050, 14.1953163
29: -2.0598311, 12.6758442, -2.0948944, 12.6698074, -11.2671661, 11.2883682
30: -8.1613245, 14.8220596, -8.2401190, 14.8164034, -20.1298294, 20.2027359
31: 0.5142393, 16.0265732, 0.4674988, 16.0276966, -14.3186951, 14.3310089
32: -22.0162525, 2.0197520, -22.0155411, 2.0555153, -18.9721069, 18.8946991
33: -39.7323990, -10.4538879, -39.7377319, -10.3664589, -21.0668144, 20.8874435
34: -33.4254150, -10.0770149, -33.4253235, -10.0203962, -17.5709076, 17.4153099
35: -24.0849648, -0.8098147, -24.0880356, -0.7850816, -18.8013306, 18.7298813
36: -20.8064423, 5.2904553, -20.8104706, 5.3022823, -20.0982742, 20.0927582
37: -32.3093185, -2.6810193, -32.3244171, -2.6722631, -25.9605560, 26.0056000
38: -28.7719898, 0.6334195, -28.7833595, 0.6483541, -24.4197159, 24.4588242
39: -43.9813538, -10.2470970, -43.9787903, -10.1578274, -23.1809158, 23.0238419
40: -31.3483429, -13.0280313, -31.3480225, -13.0184021, -12.9286385, 12.9158516
41: -19.8924065, 2.0529530, -19.8932152, 2.0754123, -18.7985764, 18.7579041
42: -20.1033878, -3.5333395, -20.1102028, -3.5141864, -13.5873871, 13.6111145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=144, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1690

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_B2_B1_B2_B2_A1

### Relational analysis result of IS_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4744640, upper bound: 10.5038322
time: 22.55 seconds

## Relational analysis of IS_B2_B1_B2_B2_A2

### Relational analysis result of IS_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4951469, upper bound: 10.5082401
time: 23.08 seconds

## BFS IS instance: IS_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -29.3696175, -2.3996496, -29.3602505, -2.2840080, -18.1458015, 18.1403008
1: -13.7459602, 2.6726954, -13.7343025, 2.7088728, -11.8195076, 11.8449326
2: -12.0359325, 4.0693426, -12.0313005, 4.1319561, -10.5986824, 10.5951939
3: -21.1464195, -0.8764973, -21.1415272, -0.7471271, -16.3350220, 16.2190742
4: -19.4623604, 2.6995010, -19.4629688, 2.7938094, -14.4451485, 14.4616661
5: -15.5482206, 4.2275991, -15.5411930, 4.3355112, -15.4173775, 15.3666801
6: -21.6596889, -0.4428911, -21.6783905, -0.4499397, -16.6921158, 16.6612778
7: -18.7501011, 2.4228663, -18.7388783, 2.4633350, -16.9172897, 16.9473953
8: -28.8884907, -1.4090915, -28.8740082, -1.3766384, -17.7537460, 17.8410606
9: -19.1447296, 2.4770775, -19.1398277, 2.6058636, -17.9084625, 17.8537140
10: -16.8663788, 5.3584595, -16.9130039, 5.4107556, -19.8925858, 19.9050407
11: -2.7222850, 15.8106737, -2.8815768, 15.7984600, -17.1223869, 17.3072662
12: -17.3374786, 13.1660528, -17.4571075, 13.1517897, -24.3029938, 24.3084030
13: -30.3700542, -1.5850267, -30.3406296, -1.4960623, -20.9628220, 20.8117256
14: -34.0955811, 0.3065500, -34.2279739, 0.3122234, -29.3093872, 29.4542694
15: -15.3558321, 5.1836262, -15.3557997, 5.2757096, -18.6845245, 18.6082230
16: -15.4941082, 6.2664647, -15.5045862, 6.3174334, -19.2059250, 19.2441788
17: -23.0478382, 1.8092773, -23.1746407, 1.8018866, -23.2237930, 23.3290176
18: 1.8144021, 23.2603531, 1.6636443, 23.2587509, -18.8237991, 18.9614563
19: -0.8266640, 11.5439968, -0.8996730, 11.5366011, -11.0436134, 11.0918865
20: -4.4593468, 9.5803528, -4.5495958, 9.5736828, -13.3239784, 13.4213829
21: -1.3846316, 15.5774393, -1.4665213, 15.5609446, -15.9554138, 16.0576591
22: -3.0756822, 11.4904690, -3.0978198, 11.4886293, -13.5617104, 13.5459175
23: -1.3620262, 15.6329823, -1.4454231, 15.6197662, -13.2259636, 13.3060036
24: -1.8916926, 16.2894917, -1.9799900, 16.2801590, -15.0339203, 15.1278610
25: -2.7094498, 16.3984451, -2.7726250, 16.3859768, -17.5024338, 17.5668716
26: -5.3855557, 21.1597595, -5.5805726, 21.1579666, -25.4163132, 25.5963058
27: -0.4454765, 15.5909901, -0.5224061, 15.5778637, -13.4583588, 13.5537338
28: -1.4844246, 15.4894314, -1.5641069, 15.4798031, -14.1440620, 14.1965637
29: -2.0526354, 12.6823711, -2.1124115, 12.6847515, -11.2491226, 11.2731209
30: -8.1535130, 14.8105755, -8.2650146, 14.7992783, -20.1005249, 20.2436295
31: 0.5240703, 16.0098228, 0.4439597, 16.0019665, -14.2977638, 14.3433838
32: -21.9714127, 2.0152469, -21.9556541, 2.0233841, -18.8958435, 18.7801590
33: -39.6612625, -10.4548979, -39.6358566, -10.3589420, -20.9923363, 20.7952919
34: -33.3720207, -10.0801220, -33.3615646, -10.0550852, -17.4813805, 17.3215485
35: -24.0361328, -0.8102987, -24.0153160, -0.7893496, -18.7418442, 18.6377945
36: -20.7586937, 5.2889266, -20.7416515, 5.2750659, -20.0197372, 19.9872589
37: -32.2493286, -2.6861639, -32.2492523, -2.7072544, -25.8664856, 25.9874878
38: -28.7353249, 0.6323223, -28.7634697, 0.6327295, -24.3581238, 24.4014664
39: -43.9198418, -10.2483902, -43.8906975, -10.1513252, -23.1094513, 22.9661713
40: -31.3096695, -13.0289488, -31.2961349, -13.0406837, -12.8634644, 12.8975601
41: -19.8570175, 2.0526586, -19.8589058, 2.0609865, -18.7474442, 18.7146454
42: -20.0808754, -3.5361538, -20.1008873, -3.5228062, -13.5626106, 13.5808640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=144, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=161, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 824

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_B2_B2_B1_B2_B1

### Relational analysis result of IS_B2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4993716, upper bound: 10.4875144
time: 24.64 seconds

## Relational analysis of IS_B2_B2_B1_B2_B2

### Relational analysis result of IS_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5037763, upper bound: 10.5082399
time: 22.63 seconds

## BFS IS instance: IS_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -29.3719063, -2.3734765, -29.3768291, -2.2376142, -18.2971115, 18.1864204
1: -13.7476492, 2.7024834, -13.7495956, 2.7610734, -11.9291496, 11.8927536
2: -12.0369644, 4.0880380, -12.0396299, 4.1646614, -10.6854553, 10.6235638
3: -21.1484184, -0.8583264, -21.1506100, -0.7133770, -16.4073486, 16.2608528
4: -19.4662209, 2.7146487, -19.4715900, 2.8206739, -14.5962448, 14.5029945
5: -15.5499582, 4.2517071, -15.5537596, 4.3792906, -15.5189667, 15.4050484
6: -21.6886692, -0.4400072, -21.7285690, -0.4276581, -16.7446060, 16.7953796
7: -18.7529392, 2.4502878, -18.7588539, 2.5125341, -17.0394897, 16.9951096
8: -28.8914757, -1.3781233, -28.8978310, -1.3236856, -17.9270554, 17.8970795
9: -19.1487217, 2.5038373, -19.1586227, 2.6531949, -18.0306320, 17.9014969
10: -16.8726635, 5.3788376, -16.9246712, 5.4488912, -19.9540100, 19.9397812
11: -2.7312136, 15.8334904, -2.9055095, 15.8390970, -17.1532936, 17.3363991
12: -17.3847771, 13.1700382, -17.5399265, 13.1875172, -24.3882828, 24.5064087
13: -30.4141102, -1.5798330, -30.4183769, -1.4619040, -21.0365982, 20.8975296
14: -34.1059494, 0.3269591, -34.2632599, 0.3504181, -29.3349762, 29.5164490
15: -15.3618364, 5.1934128, -15.3703480, 5.2936964, -18.7220230, 18.6369476
16: -15.5008068, 6.3092384, -15.5306149, 6.3913260, -19.3477249, 19.3151436
17: -23.0799084, 1.8126545, -23.2309017, 1.8311315, -23.2857819, 23.4179459
18: 1.8073254, 23.2794685, 1.6444397, 23.2922935, -18.8349762, 18.9794159
19: -0.8383942, 11.5597258, -0.9239602, 11.5639668, -11.0798378, 11.1280231
20: -4.4706039, 9.5971489, -4.5763311, 9.6036949, -13.3592300, 13.4619370
21: -1.3977957, 15.6100588, -1.5074625, 15.6178780, -16.0046844, 16.1279640
22: -3.0947506, 11.4922256, -3.1338317, 11.4975195, -13.5886116, 13.6004639
23: -1.3713312, 15.6539230, -1.4677286, 15.6562748, -13.2597618, 13.3477631
24: -1.8992577, 16.3089600, -2.0028348, 16.3149681, -15.0651016, 15.1640358
25: -2.7215314, 16.4187393, -2.8005590, 16.4210320, -17.5402756, 17.6157570
26: -5.4021692, 21.1772690, -5.6166029, 21.1890163, -25.4429703, 25.6375656
27: -0.4498000, 15.6117249, -0.5458570, 15.6143456, -13.4792328, 13.5934792
28: -1.4933448, 15.5073032, -1.5849009, 15.5107050, -14.1739922, 14.2401695
29: -2.0603538, 12.6839581, -2.1277404, 12.6879692, -11.2813110, 11.3311615
30: -8.1630507, 14.8305531, -8.2883511, 14.8359785, -20.1479416, 20.2559814
31: 0.5119805, 16.0276279, 0.4170775, 16.0328979, -14.3335838, 14.3808289
32: -22.0162239, 2.0203047, -22.0330086, 2.0591664, -18.9768372, 18.9182472
33: -39.7288933, -10.4521389, -39.7554932, -10.3204365, -21.1075859, 20.9153633
34: -33.4268341, -10.0755806, -33.4581985, -10.0123234, -17.5831718, 17.4550934
35: -24.0830040, -0.8079486, -24.0997410, -0.7594695, -18.8258896, 18.7494545
36: -20.8076229, 5.2908163, -20.8281326, 5.3066158, -20.1025162, 20.1166458
37: -32.3116455, -2.6850252, -32.3610878, -2.6740551, -25.9673538, 26.0550003
38: -28.7752495, 0.6356864, -28.8370171, 0.6592917, -24.4337463, 24.5331573
39: -43.9886703, -10.2463379, -44.0135727, -10.1158676, -23.2236557, 23.0596962
40: -31.3506012, -13.0277424, -31.3686333, -13.0147266, -12.9344368, 12.9429626
41: -19.8934994, 2.0562117, -19.9234123, 2.0874619, -18.8127975, 18.7956467
42: -20.1045971, -3.5319715, -20.1423626, -3.5053263, -13.5973015, 13.6525917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=144, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1671

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_B2_B2_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5038320, upper bound: 10.4875144
time: 18.07 seconds

## Relational analysis of IS_B2_B2_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5082398, upper bound: 10.5082399
time: 19.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 39.67 seconds
IS_B2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 39.67
Output dim: 18, lower bound: -10.4700052, upper bound: 10.5038322
IS_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 39.67
Output dim: 18, lower bound: -10.4906896, upper bound: 10.5082401
IS_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 39.67
Output dim: 18, lower bound: -10.4744640, upper bound: 10.5038322
IS_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 39.67
Output dim: 18, lower bound: -10.4951469, upper bound: 10.5082401
IS_B2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 39.67
Output dim: 18, lower bound: -10.4993716, upper bound: 10.4875144
IS_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 39.67
Output dim: 18, lower bound: -10.5037763, upper bound: 10.5082399
IS_B2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 39.67
Output dim: 18, lower bound: -10.5038320, upper bound: 10.4875144
IS_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 39.67
Output dim: 18, lower bound: -10.5082398, upper bound: 10.5082399

## BFS IS instance: IS_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -29.3549500, -2.4030876, -29.3315353, -2.3560638, -18.0563583, 18.1066017
1: -13.7396660, 2.6702476, -13.7221441, 2.6689994, -11.7715645, 11.8344765
2: -12.0263691, 4.0678654, -12.0154238, 4.0848389, -10.5386620, 10.5807266
3: -21.1294498, -0.8790417, -21.1133652, -0.8209324, -16.2420883, 16.1962662
4: -19.4458160, 2.6976843, -19.4322968, 2.7337685, -14.3677254, 14.4371452
5: -15.5339174, 4.2255468, -15.5168037, 4.2691679, -15.3271179, 15.3463783
6: -21.6581497, -0.4469280, -21.6534538, -0.4609566, -16.6805344, 16.6144981
7: -18.7353210, 2.4214249, -18.7140617, 2.4166503, -16.8576889, 16.9280243
8: -28.8748817, -1.4107876, -28.8467312, -1.4279146, -17.6689186, 17.8186111
9: -19.1274796, 2.4728284, -19.1032429, 2.5311847, -17.8157043, 17.8077888
10: -16.8610001, 5.3563795, -16.8705730, 5.3802681, -19.8388252, 19.8542671
11: -2.7185531, 15.7951584, -2.8104546, 15.7654552, -17.0913239, 17.2250557
12: -17.3343983, 13.1524925, -17.3817806, 13.1216879, -24.2701797, 24.2010498
13: -30.3593559, -1.5955300, -30.3167534, -1.5619125, -20.8849640, 20.7464790
14: -34.0855560, 0.3050985, -34.1844559, 0.2909007, -29.2764740, 29.3999557
15: -15.3495426, 5.1809402, -15.3394699, 5.2296400, -18.6222839, 18.5891724
16: -15.4832592, 6.2640414, -15.4740849, 6.2579937, -19.1356697, 19.2092628
17: -23.0410233, 1.8057160, -23.1316757, 1.7791317, -23.1958542, 23.2789688
18: 1.8208089, 23.2512283, 1.7501392, 23.2327538, -18.7889252, 18.8650475
19: -0.8240848, 11.5435638, -0.8584380, 11.5316029, -11.0296516, 11.0513420
20: -4.4540834, 9.5799751, -4.5029974, 9.5687685, -13.3086281, 13.3755417
21: -1.3801603, 15.5767002, -1.4292793, 15.5531082, -15.9368515, 16.0234222
22: -3.0726016, 11.4889545, -3.0686824, 11.4851522, -13.5424309, 13.5142899
23: -1.3585081, 15.6244717, -1.3902445, 15.6023321, -13.2080803, 13.2465591
24: -1.8881736, 16.2805710, -1.9200888, 16.2573586, -15.0027351, 15.0611229
25: -2.7064281, 16.3945427, -2.7208591, 16.3764877, -17.4896317, 17.5158615
26: -5.3793740, 21.1576557, -5.4857779, 21.1437664, -25.3857422, 25.4944382
27: -0.4399247, 15.5848637, -0.4719825, 15.5630913, -13.4403877, 13.4992409
28: -1.4808550, 15.4866228, -1.5180750, 15.4705677, -14.1249580, 14.1498528
29: -2.0499942, 12.6741095, -2.0784411, 12.6665144, -11.2209892, 11.2289734
30: -8.1482697, 14.8018847, -8.2148619, 14.7795897, -20.0671921, 20.1881332
31: 0.5281143, 16.0085926, 0.4953671, 15.9966574, -14.2765617, 14.2919846
32: -21.9709892, 2.0117140, -21.9379520, 2.0180793, -18.8887177, 18.7348022
33: -39.6638718, -10.4623013, -39.6175537, -10.4080334, -20.9475174, 20.7289581
34: -33.3695412, -10.0830059, -33.3281326, -10.0640764, -17.4695816, 17.2789154
35: -24.0371513, -0.8165202, -24.0030727, -0.8173101, -18.7138596, 18.6005554
36: -20.7566528, 5.2833633, -20.7234993, 5.2679329, -20.0122375, 19.9289093
37: -32.2460480, -2.6876764, -32.2119522, -2.7084122, -25.8556366, 25.9059906
38: -28.7307568, 0.6248446, -28.7090664, 0.6189899, -24.3400879, 24.2880096
39: -43.9114838, -10.2573242, -43.8552399, -10.1977139, -23.0611572, 22.8749847
40: -31.3069878, -13.0316658, -31.2752953, -13.0456963, -12.8561287, 12.8515015
41: -19.8555012, 2.0465596, -19.8284836, 2.0474675, -18.7312546, 18.6606445
42: -20.0793533, -3.5389380, -20.0685349, -3.5324917, -13.5493431, 13.5364799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=147, inp2_unstable=144, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=161, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 559

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4716446, upper bound: 10.5070516
time: 20.62 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4894890, upper bound: 10.5070516
time: 23.14 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -29.3572254, -2.3768983, -29.3481102, -2.3097029, -18.2076912, 18.1527328
1: -13.7413349, 2.7000463, -13.7374249, 2.7211771, -11.8811874, 11.8822861
2: -12.0274124, 4.0865555, -12.0237370, 4.1175594, -10.6254158, 10.6090698
3: -21.1314220, -0.8608222, -21.1224709, -0.7871532, -16.3144302, 16.2380486
4: -19.4496861, 2.7128468, -19.4409580, 2.7606225, -14.5188065, 14.4784584
5: -15.5356588, 4.2496557, -15.5293417, 4.3129110, -15.4287567, 15.3847466
6: -21.6871147, -0.4440088, -21.7036667, -0.4386749, -16.7330399, 16.7486000
7: -18.7381325, 2.4488492, -18.7340488, 2.4658337, -16.9798889, 16.9757614
8: -28.8778076, -1.3798180, -28.8705196, -1.3749323, -17.8422089, 17.8745728
9: -19.1314735, 2.4995742, -19.1220055, 2.5784917, -17.9378586, 17.8555794
10: -16.8672848, 5.3767366, -16.8822670, 5.4184008, -19.9002151, 19.8889923
11: -2.7274909, 15.8179760, -2.8344262, 15.8061447, -17.1222610, 17.2541122
12: -17.3817711, 13.1564503, -17.4646187, 13.1574163, -24.3554688, 24.3990631
13: -30.4034443, -1.5904055, -30.3944645, -1.5277243, -20.9587250, 20.8322296
14: -34.0959244, 0.3254809, -34.2197685, 0.3291211, -29.3020630, 29.4620819
15: -15.3555527, 5.1907253, -15.3540192, 5.2476187, -18.6596909, 18.6179008
16: -15.4899454, 6.3067989, -15.5001621, 6.3318830, -19.2774658, 19.2802887
17: -23.0730877, 1.8090854, -23.1879539, 1.8084028, -23.2578430, 23.3678513
18: 1.8137641, 23.2703362, 1.7309489, 23.2662907, -18.8000488, 18.8829956
19: -0.8358192, 11.5593042, -0.8827343, 11.5589714, -11.0658855, 11.0874481
20: -4.4653254, 9.5967808, -4.5297456, 9.5987597, -13.3438950, 13.4161072
21: -1.3933015, 15.6093464, -1.4702020, 15.6100435, -15.9860992, 16.0936852
22: -3.0917115, 11.4907255, -3.1046994, 11.4940310, -13.5693207, 13.5688324
23: -1.3678370, 15.6453972, -1.4125586, 15.6388569, -13.2418823, 13.2883110
24: -1.8957100, 16.3000507, -1.9429593, 16.2921448, -15.0339165, 15.0972633
25: -2.7184901, 16.4148521, -2.7487860, 16.4115372, -17.5275040, 17.5647659
26: -5.3959770, 21.1752052, -5.5217080, 21.1748390, -25.4122696, 25.5357056
27: -0.4442215, 15.6056108, -0.4954271, 15.5995941, -13.4612541, 13.5389252
28: -1.4897814, 15.5044880, -1.5388813, 15.5014668, -14.1548882, 14.1933937
29: -2.0577085, 12.6757126, -2.0937505, 12.6697197, -11.2531738, 11.2869797
30: -8.1578465, 14.8218393, -8.2382269, 14.8162603, -20.1145935, 20.2004700
31: 0.5160146, 16.0264111, 0.4684672, 16.0276165, -14.3123817, 14.3294182
32: -22.0158310, 2.0167418, -22.0152950, 2.0538626, -18.9697266, 18.8728714
33: -39.7315216, -10.4595604, -39.7372665, -10.3695135, -21.0627861, 20.8490562
34: -33.4243622, -10.0785055, -33.4247627, -10.0212669, -17.5714226, 17.4124413
35: -24.0839920, -0.8141308, -24.0875092, -0.7874055, -18.7978897, 18.7122383
36: -20.8056030, 5.2852554, -20.8099937, 5.2994790, -20.0950623, 20.0582809
37: -32.3083649, -2.6865463, -32.3238258, -2.6751981, -25.9564972, 25.9735565
38: -28.7706928, 0.6281657, -28.7826347, 0.6455665, -24.4157028, 24.4197693
39: -43.9803162, -10.2552919, -43.9781570, -10.1622314, -23.1753922, 22.9684601
40: -31.3479385, -13.0305300, -31.3478203, -13.0197277, -12.9271011, 12.8968925
41: -19.8920517, 2.0501387, -19.8929863, 2.0738876, -18.7966461, 18.7416458
42: -20.1030903, -3.5347698, -20.1100159, -3.5149915, -13.5840416, 13.6082115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=147, inp2_unstable=144, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 559

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_B2_B1_B2_B2_A2_A1

### Relational analysis result of IS_B2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4761052, upper bound: 10.5070516
time: 9.31 seconds

## Relational analysis of IS_B2_B1_B2_B2_A2_A2

### Relational analysis result of IS_B2_B1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4939500, upper bound: 10.5070516
time: 26.28 seconds

## BFS IS instance: IS_B2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -29.3694210, -2.4008055, -29.3598671, -2.2861814, -18.1348610, 18.1373978
1: -13.7456665, 2.6719193, -13.7337847, 2.7074211, -11.8182373, 11.8425903
2: -12.0349169, 4.0689831, -12.0294113, 4.1312952, -10.5971451, 10.5898380
3: -21.1441021, -0.8770909, -21.1375332, -0.7481508, -16.3337555, 16.2143097
4: -19.4608994, 2.6990790, -19.4603806, 2.7930856, -14.4432144, 14.4563332
5: -15.5462704, 4.2270846, -15.5380821, 4.3345919, -15.4160690, 15.3545456
6: -21.6595554, -0.4436150, -21.6780281, -0.4511089, -16.6770325, 16.6602211
7: -18.7484245, 2.4224396, -18.7357159, 2.4626002, -16.9171753, 16.9461288
8: -28.8878670, -1.4094806, -28.8730221, -1.3773556, -17.7526207, 17.8198853
9: -19.1446075, 2.4753311, -19.1396332, 2.6027205, -17.8937683, 17.8510017
10: -16.8644447, 5.3579912, -16.9096107, 5.4099059, -19.8896980, 19.8802071
11: -2.7211525, 15.8105268, -2.8795104, 15.7981777, -17.1208572, 17.3023148
12: -17.3369293, 13.1654758, -17.4562206, 13.1506948, -24.2832108, 24.3065948
13: -30.3696842, -1.5894761, -30.3400974, -1.5044689, -20.9167938, 20.8067055
14: -34.0927048, 0.3062549, -34.2232246, 0.3116779, -29.3064423, 29.4470520
15: -15.3551311, 5.1832952, -15.3546524, 5.2751222, -18.6829758, 18.5964432
16: -15.4937820, 6.2654533, -15.5040045, 6.3156157, -19.1978226, 19.2424202
17: -23.0451527, 1.8087826, -23.1696167, 1.8009577, -23.2189941, 23.3239441
18: 1.8164511, 23.2602520, 1.6674504, 23.2585068, -18.8212738, 18.9503441
19: -0.8260117, 11.5439100, -0.8985000, 11.5364361, -11.0416870, 11.0895443
20: -4.4580417, 9.5802288, -4.5471268, 9.5735092, -13.3224564, 13.4186897
21: -1.3835449, 15.5773544, -1.4645457, 15.5608311, -15.9564209, 16.0547752
22: -3.0746834, 11.4903412, -3.0959594, 11.4884567, -13.5603752, 13.5380287
23: -1.3609271, 15.6329460, -1.4434190, 15.6196690, -13.2243996, 13.3001289
24: -1.8902488, 16.2893677, -1.9773517, 16.2799587, -15.0320511, 15.1151848
25: -2.7086363, 16.3972931, -2.7711949, 16.3840790, -17.5005569, 17.5649872
26: -5.3833919, 21.1596146, -5.5765753, 21.1577091, -25.4136353, 25.5835037
27: -0.4434090, 15.5908728, -0.5185094, 15.5776472, -13.4560471, 13.5480957
28: -1.4834604, 15.4893475, -1.5623403, 15.4796352, -14.1421318, 14.1911812
29: -2.0514832, 12.6822643, -2.1102846, 12.6846313, -11.2477264, 11.2591209
30: -8.1516218, 14.8104448, -8.2615261, 14.7990589, -20.0982590, 20.2284241
31: 0.5250306, 16.0097351, 0.4457569, 16.0018177, -14.2961845, 14.3370399
32: -21.9711475, 2.0136161, -21.9552078, 2.0203676, -18.8739014, 18.7778168
33: -39.6607513, -10.4578991, -39.6349716, -10.3646069, -20.9539413, 20.7912521
34: -33.3714523, -10.0809746, -33.3605576, -10.0566044, -17.4784698, 17.3220673
35: -24.0355949, -0.8126218, -24.0143089, -0.7936554, -18.7241745, 18.6343536
36: -20.7582283, 5.2861032, -20.7407761, 5.2698374, -19.9852676, 19.9840469
37: -32.2488022, -2.6891208, -32.2482452, -2.7127304, -25.8344727, 25.9833984
38: -28.7346039, 0.6295104, -28.7621822, 0.6274652, -24.3190994, 24.3975067
39: -43.9191856, -10.2528105, -43.8895950, -10.1595316, -23.0540543, 22.9605865
40: -31.3094177, -13.0302744, -31.2957344, -13.0431776, -12.8445206, 12.8960266
41: -19.8567905, 2.0511098, -19.8585453, 2.0581787, -18.7311554, 18.7127304
42: -20.0806885, -3.5369182, -20.1005402, -3.5242326, -13.5595169, 13.5774612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=143, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=161, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 974
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 824

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_B2_B2_B1_B2_B2_A1

### Relational analysis result of IS_B2_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4847377, upper bound: 10.5070514
time: 20.57 seconds

## Relational analysis of IS_B2_B2_B1_B2_B2_A2

### Relational analysis result of IS_B2_B2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5025884, upper bound: 10.5070514
time: 22.11 seconds

## BFS IS instance: IS_B2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -29.3717175, -2.3745852, -29.3764496, -2.2398324, -18.2861633, 18.1835251
1: -13.7473564, 2.7017171, -13.7490625, 2.7596490, -11.9278564, 11.8904152
2: -12.0359440, 4.0876818, -12.0377178, 4.1640029, -10.6839256, 10.6181736
3: -21.1461048, -0.8588853, -21.1466103, -0.7143474, -16.4060860, 16.2560883
4: -19.4647999, 2.7142291, -19.4690056, 2.8199744, -14.5943222, 14.4976692
5: -15.5480165, 4.2512007, -15.5506544, 4.3783407, -15.5176697, 15.3929176
6: -21.6884861, -0.4407167, -21.7282219, -0.4287796, -16.7295609, 16.7943039
7: -18.7512627, 2.4498696, -18.7557125, 2.5117688, -17.0393677, 16.9938889
8: -28.8908100, -1.3784957, -28.8968258, -1.3243709, -17.9259491, 17.8758965
9: -19.1485901, 2.5020933, -19.1583824, 2.6500232, -18.0159302, 17.8987808
10: -16.8707104, 5.3783808, -16.9213142, 5.4480314, -19.9511337, 19.9149323
11: -2.7300956, 15.8333368, -2.9034333, 15.8388309, -17.1517792, 17.3314590
12: -17.3842754, 13.1694489, -17.5390682, 13.1864748, -24.3685226, 24.5046005
13: -30.4138050, -1.5843310, -30.4177704, -1.4702682, -20.9905624, 20.8924904
14: -34.1031036, 0.3266363, -34.2584839, 0.3499022, -29.3319397, 29.5091705
15: -15.3611469, 5.1931200, -15.3691816, 5.2931275, -18.7204590, 18.6251755
16: -15.5004711, 6.3082128, -15.5300341, 6.3894963, -19.3396645, 19.3134041
17: -23.0771828, 1.8121772, -23.2259235, 1.8302214, -23.2809525, 23.4128799
18: 1.8094063, 23.2793293, 1.6482358, 23.2920609, -18.8324509, 18.9683189
19: -0.8377542, 11.5596323, -0.9227896, 11.5638018, -11.0779190, 11.1256866
20: -4.4692864, 9.5970221, -4.5738659, 9.6035099, -13.3577194, 13.4592571
21: -1.3967166, 15.6099815, -1.5054626, 15.6177645, -16.0057068, 16.1250687
22: -3.0937965, 11.4921179, -3.1320016, 11.4973469, -13.5872803, 13.5925980
23: -1.3702202, 15.6538868, -1.4657345, 15.6561861, -13.2582054, 13.3418713
24: -1.8978109, 16.3088493, -2.0001926, 16.3147278, -15.0632362, 15.1513481
25: -2.7207379, 16.4175758, -2.7991042, 16.4191284, -17.5384216, 17.6138840
26: -5.3999834, 21.1771374, -5.6125698, 21.1887741, -25.4402695, 25.6247177
27: -0.4477568, 15.6116161, -0.5419717, 15.6141510, -13.4769363, 13.5878220
28: -1.4923668, 15.5072117, -1.5831161, 15.5105610, -14.1720734, 14.2347603
29: -2.0591986, 12.6838894, -2.1256154, 12.6878405, -11.2799034, 11.3171349
30: -8.1611929, 14.8303919, -8.2848568, 14.8357210, -20.1456909, 20.2407608
31: 0.5129085, 16.0275421, 0.4188509, 16.0327568, -14.3319969, 14.3744659
32: -22.0159988, 2.0186286, -22.0325851, 2.0561690, -18.9549103, 18.9158554
33: -39.7283554, -10.4551382, -39.7546082, -10.3260288, -21.0691833, 20.9113235
34: -33.4263000, -10.0764799, -33.4571800, -10.0138292, -17.5802841, 17.4556007
35: -24.0824680, -0.8102493, -24.0987644, -0.7637668, -18.8082581, 18.7460251
36: -20.8072128, 5.2880354, -20.8273354, 5.3014240, -20.0680542, 20.1134186
37: -32.3111191, -2.6879334, -32.3601532, -2.6795292, -25.9352875, 26.0509796
38: -28.7745304, 0.6329017, -28.8357430, 0.6539755, -24.3946838, 24.5292282
39: -43.9881058, -10.2507153, -44.0124512, -10.1240587, -23.1682663, 23.0541382
40: -31.3503304, -13.0291061, -31.3682613, -13.0171909, -12.9155045, 12.9414482
41: -19.8933239, 2.0547130, -19.9230499, 2.0846047, -18.7965469, 18.7937393
42: -20.1044273, -3.5327888, -20.1420612, -3.5067506, -13.5941925, 13.6492157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=148, inp2_unstable=143, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1671

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_B2_B2_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4892012, upper bound: 10.5070514
time: 26.61 seconds

## Relational analysis of IS_B2_B2_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5070512, upper bound: 10.5070514
time: 23.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 52.60 seconds
IS_B2_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 52.60
Output dim: 18, lower bound: -10.4716446, upper bound: 10.5070516
IS_B2_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 52.60
Output dim: 18, lower bound: -10.4894890, upper bound: 10.5070516
IS_B2_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 52.60
Output dim: 18, lower bound: -10.4761052, upper bound: 10.5070516
IS_B2_B1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 52.60
Output dim: 18, lower bound: -10.4939500, upper bound: 10.5070516
IS_B2_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 52.60
Output dim: 18, lower bound: -10.4847377, upper bound: 10.5070514
IS_B2_B2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 52.60
Output dim: 18, lower bound: -10.5025884, upper bound: 10.5070514
IS_B2_B2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 52.60
Output dim: 18, lower bound: -10.4892012, upper bound: 10.5070514
IS_B2_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 52.60
Output dim: 18, lower bound: -10.5070512, upper bound: 10.5070514

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 32.99 + 1169.73 = 1202.72 seconds
