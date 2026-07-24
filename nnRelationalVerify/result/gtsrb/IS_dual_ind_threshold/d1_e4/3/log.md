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
execution time: IAR + RelationalAnalysis = 2.78 + 30.37 = 33.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 18, lower bound: -10.5176063, upper bound: 10.5176063

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5140667, upper bound: 10.4928949
time: 28.95 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5140667, upper bound: 10.5160989
time: 21.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 50.19 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 50.19
Output dim: 18, lower bound: -10.5140667, upper bound: 10.4928949
IS_A2, status: Status.UNKNOWN, split count: 1, time: 50.19
Output dim: 18, lower bound: -10.5140667, upper bound: 10.5160989

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -29.3212452, -2.3847294, -29.3456249, -2.3777218, -18.1341972, 18.1541176
1: -13.7399054, 2.6999366, -13.7447424, 2.7025986, -11.8850479, 11.8881149
2: -12.0200615, 4.0828757, -12.0287380, 4.0865593, -10.6057167, 10.6119099
3: -21.0944576, -0.8682437, -21.1200085, -0.8613391, -16.2107391, 16.2301636
4: -19.4313202, 2.7068806, -19.4481277, 2.7122545, -14.4611740, 14.4760551
5: -15.5004539, 4.2413936, -15.5241718, 4.2477551, -15.3601646, 15.3778229
6: -21.6798401, -0.4533482, -21.6853123, -0.4426556, -16.7217636, 16.7110023
7: -18.7244091, 2.4421372, -18.7390289, 2.4473591, -16.9642715, 16.9739532
8: -28.8724747, -1.3800669, -28.8831196, -1.3775148, -17.8800201, 17.8878517
9: -19.0932732, 2.4921360, -19.1194401, 2.5004084, -17.8425064, 17.8622131
10: -16.8505096, 5.3660316, -16.8627148, 5.3741026, -19.8580856, 19.8625298
11: -2.7111146, 15.7773027, -2.7224462, 15.8027973, -17.1351509, 17.1158714
12: -17.3704071, 13.0845251, -17.3790855, 13.1249733, -24.3185730, 24.2855072
13: -30.3874607, -1.5994458, -30.4053497, -1.5872488, -20.9126282, 20.9189568
14: -34.0716896, 0.2459955, -34.0906219, 0.2827811, -29.2742157, 29.2529526
15: -15.3186750, 5.1788816, -15.3408775, 5.1873083, -18.5900345, 18.6025238
16: -15.4849110, 6.3019667, -15.4941597, 6.3077049, -19.2641144, 19.2666130
17: -23.0667324, 1.7052493, -23.0755577, 1.7552001, -23.2075806, 23.1626205
18: 1.8215570, 23.2271061, 1.8113050, 23.2518997, -18.7942886, 18.7802391
19: -0.8266978, 11.5565281, -0.8337603, 11.5590076, -11.0403500, 11.0444584
20: -4.4518509, 9.5780163, -4.4625025, 9.5874872, -13.3338890, 13.3358593
21: -1.3792295, 15.5836792, -1.3895230, 15.5958300, -15.9970360, 15.9969482
22: -3.0819280, 11.4813833, -3.0899014, 11.4870462, -13.5573387, 13.5597878
23: -1.3632669, 15.6528902, -1.3689761, 15.6543980, -13.2580299, 13.2605324
24: -1.8902588, 16.2933502, -1.8971596, 16.3024921, -15.0623016, 15.0603294
25: -2.7067685, 16.4130478, -2.7161498, 16.4174919, -17.5281982, 17.5314751
26: -5.3843389, 21.1198082, -5.3964763, 21.1494904, -25.3841095, 25.3669510
27: -0.4308677, 15.5759802, -0.4422579, 15.5956860, -13.4725609, 13.4631920
28: -1.4820790, 15.4965506, -1.4889598, 15.5026646, -14.1521187, 14.1524544
29: -2.0531573, 12.6639595, -2.0577085, 12.6746883, -11.2587204, 11.2512856
30: -8.1462660, 14.7826881, -8.1564083, 14.8040104, -20.1100616, 20.0961151
31: 0.5292230, 16.0185814, 0.5194664, 16.0232544, -14.2888794, 14.2940292
32: -21.9959068, 2.0087729, -22.0077553, 2.0162935, -18.9214401, 18.9249725
33: -39.6800804, -10.4705591, -39.7080612, -10.4601593, -20.9312134, 20.9523697
34: -33.3860664, -10.0871925, -33.4065018, -10.0804005, -17.4731865, 17.4853897
35: -24.0602589, -0.8187640, -24.0760918, -0.8120942, -18.7535400, 18.7607498
36: -20.7957840, 5.2812610, -20.8037491, 5.2873521, -20.0970230, 20.0928116
37: -32.2864227, -2.6988277, -32.3009262, -2.6876197, -25.9629669, 25.9503860
38: -28.7580566, 0.6231661, -28.7684383, 0.6321559, -24.4116745, 24.4041443
39: -43.9415436, -10.2574930, -43.9663925, -10.2503929, -23.0834427, 23.1005058
40: -31.3293991, -13.0389166, -31.3413429, -13.0332279, -12.9235687, 12.9196625
41: -19.8745766, 2.0449114, -19.8846970, 2.0515866, -18.7608032, 18.7635345
42: -20.0893097, -3.5424228, -20.0973625, -3.5358217, -13.5613823, 13.5592270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=147, inp2_unstable=148, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5079954, upper bound: 10.4650733
time: 26.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5128896, upper bound: 10.4917163
time: 18.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -29.3781204, -2.2991886, -29.3740978, -2.3698292, -18.1995087, 18.2761497
1: -13.7508774, 2.7292869, -13.7491512, 2.7053847, -11.9004250, 11.9236069
2: -12.0410233, 4.1238146, -12.0383110, 4.0903001, -10.6307526, 10.6664581
3: -21.1523895, -0.7773037, -21.1507854, -0.8548102, -16.2655106, 16.3537598
4: -19.4714546, 2.7703457, -19.4683609, 2.7185516, -14.5070648, 14.5654449
5: -15.5558128, 4.3208408, -15.5523891, 4.2545581, -15.4151344, 15.4868393
6: -21.7116566, -0.4288092, -21.6915474, -0.4377804, -16.7947540, 16.7485657
7: -18.7613945, 2.4725957, -18.7554951, 2.4529166, -17.0042114, 17.0227814
8: -28.8997326, -1.3671265, -28.8947105, -1.3752656, -17.9092789, 17.9149857
9: -19.1571579, 2.5907707, -19.1510773, 2.5083821, -17.9103699, 17.9900322
10: -16.8922043, 5.4264355, -16.8750572, 5.3817797, -19.9090500, 19.9406090
11: -2.8462257, 15.8383799, -2.7354178, 15.8362942, -17.2961349, 17.1815491
12: -17.4768791, 13.1859970, -17.3888378, 13.1729717, -24.4727097, 24.3934097
13: -30.4200153, -1.5102386, -30.4181633, -1.5751452, -20.9557037, 21.0142632
14: -34.2411499, 0.3355913, -34.1109238, 0.3292418, -29.4922333, 29.3491211
15: -15.3677902, 5.2575226, -15.3639622, 5.1965041, -18.6380234, 18.6917152
16: -15.5258217, 6.3410034, -15.5045757, 6.3128281, -19.3179893, 19.3255386
17: -23.2014313, 1.8210914, -23.0840206, 1.8167305, -23.4063339, 23.2837753
18: 1.7152677, 23.2891388, 1.8026648, 23.2831345, -18.9235153, 18.8373413
19: -0.8911114, 11.5635271, -0.8411636, 11.5619812, -11.1009369, 11.0804253
20: -4.5425515, 9.6017342, -4.4740124, 9.5985193, -13.4316368, 13.3648300
21: -1.4814963, 15.6140327, -1.4008512, 15.6115360, -16.1092834, 16.0276909
22: -3.1125739, 11.4991322, -3.0971920, 11.4936771, -13.5943642, 13.5879478
23: -1.4227824, 15.6570320, -1.3747120, 15.6559944, -13.3210678, 13.2805405
24: -1.9531789, 16.3155003, -1.9031072, 16.3130798, -15.1326904, 15.0837364
25: -2.7590098, 16.4233437, -2.7250676, 16.4226952, -17.5866089, 17.5513115
26: -5.5382943, 21.1919785, -5.4077749, 21.1849098, -25.5749741, 25.4420471
27: -0.5070195, 15.6138573, -0.4531560, 15.6136169, -13.5606728, 13.5055428
28: -1.5486660, 15.5102568, -1.4962897, 15.5093756, -14.2212563, 14.1837654
29: -2.1000764, 12.6869259, -2.0625851, 12.6853275, -11.3216324, 11.2769127
30: -8.2515755, 14.8353081, -8.1675377, 14.8324537, -20.2428284, 20.1550903
31: 0.4576993, 16.0325966, 0.5089359, 16.0292950, -14.3485603, 14.3360214
32: -22.0255013, 2.0626631, -22.0215416, 2.0236840, -18.9513321, 18.9873924
33: -39.7528152, -10.3584948, -39.7408600, -10.4495325, -21.0018387, 21.0959015
34: -33.4346008, -10.0148869, -33.4302101, -10.0739527, -17.5160255, 17.5853539
35: -24.0961571, -0.7781491, -24.0893993, -0.8060138, -18.7965469, 18.8126526
36: -20.8175735, 5.3094654, -20.8101883, 5.2943721, -20.1416245, 20.1215897
37: -32.3376503, -2.6628942, -32.3165932, -2.6760654, -26.0781555, 25.9880219
38: -28.7959595, 0.6616683, -28.7788715, 0.6408024, -24.4984131, 24.4536362
39: -44.0047684, -10.1494942, -43.9958801, -10.2427826, -23.1417618, 23.2309532
40: -31.3596535, -13.0164890, -31.3546829, -13.0269814, -12.9875793, 12.9530945
41: -19.8999729, 2.0858097, -19.8960190, 2.0587792, -18.7972183, 18.8150864
42: -20.1175461, -3.5068622, -20.1074181, -3.5293195, -13.6336632, 13.5968361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=147, inp2_unstable=148, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5100558, upper bound: 10.4882992
time: 25.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5149297, upper bound: 10.5149292
time: 27.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 54.99 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 54.99
Output dim: 18, lower bound: -10.5079954, upper bound: 10.4650733
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 54.99
Output dim: 18, lower bound: -10.5128896, upper bound: 10.4917163
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 54.99
Output dim: 18, lower bound: -10.5100558, upper bound: 10.4882992
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 54.99
Output dim: 18, lower bound: -10.5149297, upper bound: 10.5149292

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -29.3163872, -2.4412479, -29.3140011, -2.4803324, -18.0246429, 18.0636826
1: -13.7360134, 2.6654751, -13.7281542, 2.6392717, -11.8173637, 11.8378944
2: -12.0177193, 4.0505466, -12.0170298, 4.0271101, -10.5419998, 10.5664253
3: -21.0907459, -0.9098620, -21.1060905, -0.9383850, -16.1272316, 16.1727104
4: -19.4263458, 2.6636519, -19.4193420, 2.6337919, -14.3744125, 14.3992882
5: -15.4967432, 4.1977835, -15.5067024, 4.1680841, -15.2756500, 15.3169823
6: -21.6673069, -0.4647522, -21.6599197, -0.4653111, -16.6832809, 16.6722527
7: -18.7208729, 2.4092937, -18.7198982, 2.3869958, -16.8983688, 16.9246368
8: -28.8645744, -1.4071264, -28.8600788, -1.4284534, -17.8245163, 17.8418350
9: -19.0870743, 2.4366558, -19.0853939, 2.4015617, -17.7342911, 17.7677841
10: -16.8040085, 5.3496995, -16.7798061, 5.3418188, -19.7757568, 19.7581024
11: -2.6614730, 15.7716064, -2.6318529, 15.7753553, -17.0514641, 17.0162277
12: -17.3363895, 13.0732136, -17.3156052, 13.1006165, -24.2559586, 24.2079239
13: -30.3788280, -1.6974683, -30.3527756, -1.7632041, -20.7298203, 20.7696152
14: -34.0283279, 0.2302456, -34.0082321, 0.2540259, -29.1979828, 29.1502228
15: -15.3069525, 5.1402931, -15.3143883, 5.1166134, -18.5054779, 18.5348625
16: -15.4732580, 6.2552843, -15.4600115, 6.2235761, -19.1678619, 19.1867561
17: -23.0315762, 1.6873286, -23.0107288, 1.7173934, -23.1313934, 23.0753632
18: 1.8880820, 23.2201500, 1.9319267, 23.2251625, -18.6948738, 18.6479454
19: -0.7912912, 11.5538158, -0.7678542, 11.5525141, -10.9908276, 10.9682388
20: -4.4040279, 9.5750923, -4.3734665, 9.5807800, -13.2789307, 13.2431755
21: -1.3374929, 15.5815926, -1.3116255, 15.5909986, -15.9475937, 15.9155807
22: -3.0448375, 11.4778814, -3.0219197, 11.4747677, -13.5054512, 13.4845390
23: -1.3124094, 15.6506195, -1.2762294, 15.6354733, -13.1842804, 13.1618843
24: -1.8386497, 16.2888908, -1.8019094, 16.2803555, -14.9840775, 14.9576225
25: -2.6610489, 16.4095039, -2.6303868, 16.4089470, -17.4679260, 17.4387512
26: -5.3104649, 21.1125526, -5.2604799, 21.1292458, -25.2832870, 25.2176056
27: -0.3797388, 15.5714941, -0.3495541, 15.5733910, -13.3984032, 13.3678970
28: -1.4350777, 15.4934654, -1.4024954, 15.4858665, -14.0853500, 14.0601921
29: -2.0199792, 12.6609135, -1.9977608, 12.6562862, -11.2008400, 11.1822624
30: -8.0925770, 14.7787914, -8.0578423, 14.7760315, -20.0241013, 19.9921722
31: 0.5771046, 16.0161209, 0.6083488, 16.0148830, -14.2283058, 14.1991615
32: -21.9841175, 1.9893036, -21.9833717, 1.9792013, -18.8682327, 18.8745422
33: -39.6587906, -10.5200043, -39.6560059, -10.5485516, -20.8174133, 20.8435211
34: -33.3466797, -10.0948257, -33.3351135, -10.1044922, -17.4061356, 17.4023781
35: -24.0458431, -0.8525646, -24.0451279, -0.8744524, -18.6760406, 18.6926155
36: -20.7838993, 5.2428594, -20.7704964, 5.2183342, -20.0184326, 20.0236588
37: -32.2659683, -2.7402210, -32.2557220, -2.7609930, -25.8667526, 25.8606186
38: -28.7371960, 0.5895762, -28.7239647, 0.5724735, -24.3313675, 24.3276825
39: -43.9181099, -10.3346710, -43.8892746, -10.3872166, -22.9172363, 22.9370804
40: -31.3160973, -13.0612602, -31.3083725, -13.0723715, -12.8671379, 12.8587265
41: -19.8661861, 2.0280538, -19.8676872, 2.0188568, -18.7133255, 18.7218475
42: -20.0677681, -3.5499659, -20.0568848, -3.5523133, -13.5101662, 13.5005875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=147, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5039856, upper bound: 10.4493261
time: 21.41 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5039856, upper bound: 10.4611604
time: 27.73 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -29.3210297, -2.3864441, -29.3451729, -2.3812475, -18.1077881, 18.1516914
1: -13.7396832, 2.6987662, -13.7443371, 2.7002575, -11.8717117, 11.8864403
2: -12.0198669, 4.0818229, -12.0283785, 4.0844183, -10.5948105, 10.6104374
3: -21.0941200, -0.8696184, -21.1193619, -0.8641419, -16.1996765, 16.2279968
4: -19.4310760, 2.7054362, -19.4476509, 2.7093258, -14.4400558, 14.4738197
5: -15.5000963, 4.2400184, -15.5234518, 4.2450275, -15.3472595, 15.3757057
6: -21.6788540, -0.4545012, -21.6832638, -0.4449358, -16.7194519, 16.7073288
7: -18.7240582, 2.4411287, -18.7383156, 2.4453001, -16.9488831, 16.9720917
8: -28.8719883, -1.3815494, -28.8820724, -1.3805170, -17.8681870, 17.8848534
9: -19.0929756, 2.4903214, -19.1188622, 2.4968231, -17.8187866, 17.8596497
10: -16.8492470, 5.3652096, -16.8603973, 5.3725767, -19.8540382, 19.8542290
11: -2.7093334, 15.7770214, -2.7188809, 15.8022137, -17.1325455, 17.0980530
12: -17.3687935, 13.0838642, -17.3757515, 13.1237030, -24.3146973, 24.2768326
13: -30.3870163, -1.6022801, -30.4044571, -1.5932517, -20.8832855, 20.9152031
14: -34.0698318, 0.2445564, -34.0868759, 0.2799196, -29.2686310, 29.2457199
15: -15.3181534, 5.1774707, -15.3398619, 5.1844859, -18.5847855, 18.5998802
16: -15.4840946, 6.3004642, -15.4926023, 6.3046799, -19.2482681, 19.2634087
17: -23.0654335, 1.7045865, -23.0729942, 1.7538466, -23.2045441, 23.1518631
18: 1.8236508, 23.2267075, 1.8156338, 23.2511501, -18.7912445, 18.7661934
19: -0.8253450, 11.5562210, -0.8311491, 11.5583887, -11.0375404, 11.0398598
20: -4.4501867, 9.5778656, -4.4591818, 9.5871973, -13.3322487, 13.3322315
21: -1.3776851, 15.5835505, -1.3864937, 15.5955582, -15.9949341, 15.9914589
22: -3.0807240, 11.4810810, -3.0874488, 11.4864645, -13.5553741, 13.5511017
23: -1.3615761, 15.6526604, -1.3656049, 15.6539764, -13.2557297, 13.2445011
24: -1.8884921, 16.2930355, -1.8936195, 16.3018761, -15.0597382, 15.0435333
25: -2.7051563, 16.4128094, -2.7129536, 16.4170151, -17.5257950, 17.5226822
26: -5.3818417, 21.1193981, -5.3914952, 21.1486893, -25.3804169, 25.3548203
27: -0.4292231, 15.5757999, -0.4388747, 15.5952644, -13.4704514, 13.4571190
28: -1.4804807, 15.4963951, -1.4857483, 15.5022869, -14.1499557, 14.1366539
29: -2.0520711, 12.6637697, -2.0555444, 12.6743393, -11.2570343, 11.2323723
30: -8.1441898, 14.7825108, -8.1522942, 14.8036041, -20.1073837, 20.0738068
31: 0.5308428, 16.0184364, 0.5226517, 16.0229683, -14.2866974, 14.2866936
32: -21.9951706, 2.0075612, -22.0062561, 2.0138721, -18.9151764, 18.9211273
33: -39.6792221, -10.4715824, -39.7064056, -10.4622316, -20.9136543, 20.9488678
34: -33.3848495, -10.0876522, -33.4039116, -10.0812426, -17.4709053, 17.4698372
35: -24.0596466, -0.8193510, -24.0749073, -0.8132386, -18.7509460, 18.7602425
36: -20.7952194, 5.2800436, -20.8025932, 5.2849932, -20.0830002, 20.0902100
37: -32.2854385, -2.7001567, -32.2989464, -2.6903362, -25.9497833, 25.9461365
38: -28.7571945, 0.6222839, -28.7667103, 0.6303558, -24.4003067, 24.4003601
39: -43.9406357, -10.2597647, -43.9646683, -10.2549858, -23.0463638, 23.0960884
40: -31.3287964, -13.0394268, -31.3400307, -13.0341759, -12.9073639, 12.9175491
41: -19.8741264, 2.0439894, -19.8837357, 2.0496733, -18.7566910, 18.7607880
42: -20.0883789, -3.5428648, -20.0955582, -3.5367069, -13.5635338, 13.5560150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=147, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5096145, upper bound: 10.4752886
time: 21.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5096145, upper bound: 10.4884318
time: 23.63 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -29.3732185, -2.3557787, -29.3424721, -2.4724236, -18.0899582, 18.1857071
1: -13.7469845, 2.6948144, -13.7325325, 2.6420512, -11.8327293, 11.8733559
2: -12.0386620, 4.0914364, -12.0265942, 4.0309043, -10.5670166, 10.6209564
3: -21.1486588, -0.8189778, -21.1368790, -0.9318347, -16.1820068, 16.2962799
4: -19.4664688, 2.7271738, -19.4395161, 2.6401229, -14.4203110, 14.4886589
5: -15.5520859, 4.2772336, -15.5349331, 4.1748714, -15.3306046, 15.4259911
6: -21.6991501, -0.4402390, -21.6661472, -0.4604645, -16.7562714, 16.7097778
7: -18.7578182, 2.4397626, -18.7363281, 2.3925557, -16.9382858, 16.9734573
8: -28.8917866, -1.3941765, -28.8716850, -1.4261713, -17.8538208, 17.8689384
9: -19.1509972, 2.5353646, -19.1170406, 2.4095311, -17.8021049, 17.8955994
10: -16.8457794, 5.4101591, -16.7921772, 5.3494883, -19.8267822, 19.8362198
11: -2.7965400, 15.8326912, -2.6448231, 15.8088846, -17.2124138, 17.0819092
12: -17.4428730, 13.1747093, -17.3253269, 13.1486588, -24.4101105, 24.3158264
13: -30.4114208, -1.6083126, -30.3656406, -1.7510133, -20.7728729, 20.8648949
14: -34.1977310, 0.3198929, -34.0285416, 0.3005095, -29.4159088, 29.2464447
15: -15.3560400, 5.2189136, -15.3374929, 5.1258311, -18.5533829, 18.6239853
16: -15.5141602, 6.2943168, -15.4703856, 6.2286801, -19.2217560, 19.2456245
17: -23.1662407, 1.8031743, -23.0192261, 1.7789640, -23.3300629, 23.1964493
18: 1.7818518, 23.2822304, 1.9232702, 23.2563915, -18.8240433, 18.7050781
19: -0.8556547, 11.5608006, -0.7752714, 11.5555124, -11.0513935, 11.0042076
20: -4.4946542, 9.5988255, -4.3850060, 9.5918236, -13.3766098, 13.2721367
21: -1.4397216, 15.6119480, -1.3229570, 15.6067352, -16.0597878, 15.9463387
22: -3.0754859, 11.4956322, -3.0292561, 11.4814062, -13.5424995, 13.5126534
23: -1.3718443, 15.6547832, -1.2819929, 15.6370449, -13.2472725, 13.1818943
24: -1.9015412, 16.3110466, -1.8078160, 16.2909393, -15.0544548, 14.9810257
25: -2.7131786, 16.4197807, -2.6393404, 16.4141731, -17.5262985, 17.4585724
26: -5.4643703, 21.1847496, -5.2717500, 21.1646461, -25.4740753, 25.2927094
27: -0.4559197, 15.6093798, -0.3604393, 15.5913363, -13.4865265, 13.4102249
28: -1.5016227, 15.5071564, -1.4098473, 15.4925900, -14.1543732, 14.0915375
29: -2.0669315, 12.6838970, -2.0026400, 12.6669521, -11.2637291, 11.2078705
30: -8.1978359, 14.8313866, -8.0689240, 14.8045521, -20.1568298, 20.0511475
31: 0.5056534, 16.0301456, 0.5978293, 16.0209312, -14.2878761, 14.2411461
32: -22.0137634, 2.0431709, -21.9971943, 1.9865828, -18.8981094, 18.9369965
33: -39.7315636, -10.4079399, -39.6888237, -10.5379028, -20.8880882, 20.9870605
34: -33.3951874, -10.0225353, -33.3588028, -10.0980740, -17.4489594, 17.5023117
35: -24.0817642, -0.8119321, -24.0584183, -0.8683331, -18.7190781, 18.7444000
36: -20.8056393, 5.2711186, -20.7770119, 5.2253757, -20.0630493, 20.0524292
37: -32.3172226, -2.7042985, -32.2714005, -2.7494221, -25.9819489, 25.8981171
38: -28.7751846, 0.6280570, -28.7344360, 0.5811362, -24.4181976, 24.3772202
39: -43.9813309, -10.2266722, -43.9187393, -10.3795958, -22.9755783, 23.0674973
40: -31.3463173, -13.0388498, -31.3216991, -13.0660839, -12.9311028, 12.8921318
41: -19.8915348, 2.0688963, -19.8790455, 2.0260828, -18.7497101, 18.7733536
42: -20.0959740, -3.5144396, -20.0669346, -3.5457983, -13.5824051, 13.5382156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=147, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5060186, upper bound: 10.4493261
time: 46.11 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5060186, upper bound: 10.4843029
time: 21.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -29.3778610, -2.3009434, -29.3736458, -2.3733263, -18.1730881, 18.2737236
1: -13.7506361, 2.7281351, -13.7487316, 2.7030287, -11.8870888, 11.9219246
2: -12.0408287, 4.1227589, -12.0379372, 4.0882006, -10.6198349, 10.6649818
3: -21.1520615, -0.7786865, -21.1501465, -0.8576078, -16.2544250, 16.3515854
4: -19.4712200, 2.7688890, -19.4678230, 2.7156749, -14.4859428, 14.5632286
5: -15.5554333, 4.3194680, -15.5516329, 4.2518177, -15.4022217, 15.4847527
6: -21.7106495, -0.4299488, -21.6895275, -0.4400482, -16.7924194, 16.7448883
7: -18.7610378, 2.4716058, -18.7547512, 2.4508781, -16.9888382, 17.0209351
8: -28.8991699, -1.3685780, -28.8936634, -1.3782372, -17.8974686, 17.9119911
9: -19.1568775, 2.5889916, -19.1505013, 2.5047197, -17.8865967, 17.9874420
10: -16.8909550, 5.4256563, -16.8727360, 5.3802242, -19.9050179, 19.9322891
11: -2.8444588, 15.8381062, -2.7318249, 15.8357410, -17.2935715, 17.1637230
12: -17.4752998, 13.1854305, -17.3854866, 13.1716824, -24.4688416, 24.3847427
13: -30.4195728, -1.5131388, -30.4172630, -1.5810742, -20.9263458, 21.0104523
14: -34.2393112, 0.3341727, -34.1071548, 0.3263121, -29.4866333, 29.3419189
15: -15.3672905, 5.2560959, -15.3629465, 5.1936612, -18.6327667, 18.6890984
16: -15.5250158, 6.3395042, -15.5030107, 6.3097916, -19.3021469, 19.3223419
17: -23.2001419, 1.8204415, -23.0814323, 1.8153861, -23.4032898, 23.2730637
18: 1.7173862, 23.2887344, 1.8069830, 23.2823467, -18.9204102, 18.8232727
19: -0.8897657, 11.5632238, -0.8385282, 11.5613775, -11.0981274, 11.0758228
20: -4.5408783, 9.6015892, -4.4706883, 9.5982552, -13.4299850, 13.3611984
21: -1.4799652, 15.6139107, -1.3978109, 15.6112728, -16.1071815, 16.0222015
22: -3.1113675, 11.4988384, -3.0947537, 11.4931011, -13.5924187, 13.5792580
23: -1.4210625, 15.6568022, -1.3713212, 15.6555738, -13.3187675, 13.2644939
24: -1.9514279, 16.3152084, -1.8995218, 16.3124485, -15.1301956, 15.0669250
25: -2.7573853, 16.4231262, -2.7218752, 16.4222679, -17.5842209, 17.5425415
26: -5.5358295, 21.1915703, -5.4027925, 21.1840878, -25.5712738, 25.4298706
27: -0.5053596, 15.6136417, -0.4497561, 15.6132183, -13.5585365, 13.4994583
28: -1.5470490, 15.5100784, -1.4930925, 15.5090122, -14.2190742, 14.1679573
29: -2.0990226, 12.6867542, -2.0604360, 12.6849651, -11.3199272, 11.2579918
30: -8.2495270, 14.8351145, -8.1634350, 14.8320837, -20.2401810, 20.1327286
31: 0.4593296, 16.0324574, 0.5121703, 16.0289822, -14.3463593, 14.3286629
32: -22.0247593, 2.0614624, -22.0200119, 2.0212893, -18.9450607, 18.9835663
33: -39.7519913, -10.3594990, -39.7392273, -10.4516335, -20.9842606, 21.0923920
34: -33.4333267, -10.0152826, -33.4276047, -10.0747843, -17.5137482, 17.5697975
35: -24.0955391, -0.7787604, -24.0881653, -0.8071432, -18.7939758, 18.8121338
36: -20.8169804, 5.3082371, -20.8091354, 5.2919807, -20.1275787, 20.1189804
37: -32.3366394, -2.6642365, -32.3146133, -2.6787782, -26.0650787, 25.9837265
38: -28.7951317, 0.6607447, -28.7771435, 0.6389985, -24.4870911, 24.4498596
39: -44.0038223, -10.1517467, -43.9940643, -10.2473316, -23.1046371, 23.2265472
40: -31.3589821, -13.0169764, -31.3533669, -13.0278912, -12.9713745, 12.9509659
41: -19.8995018, 2.0849037, -19.8950634, 2.0569096, -18.7931366, 18.8123169
42: -20.1165962, -3.5072904, -20.1055984, -3.5302134, -13.6358414, 13.5936317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=147, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5117014, upper bound: 10.4985757
time: 19.26 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5117014, upper bound: 10.5117008
time: 18.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 39.76 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 39.76
Output dim: 18, lower bound: -10.5039856, upper bound: 10.4493261
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 39.76
Output dim: 18, lower bound: -10.5039856, upper bound: 10.4611604
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 39.76
Output dim: 18, lower bound: -10.5096145, upper bound: 10.4752886
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 39.76
Output dim: 18, lower bound: -10.5096145, upper bound: 10.4884318
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 39.76
Output dim: 18, lower bound: -10.5060186, upper bound: 10.4493261
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 39.76
Output dim: 18, lower bound: -10.5060186, upper bound: 10.4843029
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 39.76
Output dim: 18, lower bound: -10.5117014, upper bound: 10.4985757
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 39.76
Output dim: 18, lower bound: -10.5117014, upper bound: 10.5117008

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -29.2920799, -2.3903637, -29.3290997, -2.3834882, -18.0756683, 18.1309853
1: -13.7271042, 2.6957469, -13.7373199, 2.6985948, -11.8573685, 11.8762779
2: -12.0042315, 4.0792804, -12.0196972, 4.0830026, -10.5769615, 10.5988731
3: -21.0675831, -0.8747945, -21.1045399, -0.8670812, -16.1697464, 16.2076874
4: -19.4029732, 2.7019005, -19.4320488, 2.7073727, -14.4119987, 14.4558792
5: -15.4768391, 4.2365150, -15.5105362, 4.2430649, -15.3215332, 15.3589897
6: -21.6750832, -0.4611893, -21.6811523, -0.4486904, -16.7109909, 16.6965103
7: -18.6998024, 2.4385805, -18.7248268, 2.4438930, -16.9234314, 16.9561768
8: -28.8450127, -1.3846350, -28.8670540, -1.3822527, -17.8394165, 17.8660278
9: -19.0589256, 2.4856141, -19.0999756, 2.4941266, -17.7821884, 17.8364601
10: -16.8442345, 5.3612480, -16.8575630, 5.3703251, -19.8472290, 19.8476257
11: -2.7030029, 15.7469873, -2.7152622, 15.7851944, -17.1116791, 17.0671959
12: -17.3633461, 13.0579062, -17.3726845, 13.1092787, -24.2938232, 24.2468643
13: -30.3652611, -1.6085482, -30.3921280, -1.5968113, -20.8563309, 20.8957214
14: -34.0562592, 0.2425652, -34.0792732, 0.2787914, -29.2514267, 29.2338791
15: -15.3065624, 5.1723204, -15.3334045, 5.1815987, -18.5695267, 18.5876503
16: -15.4611063, 6.2982073, -15.4797668, 6.3033962, -19.2240868, 19.2481384
17: -23.0591755, 1.6942616, -23.0694561, 1.7480383, -23.1910782, 23.1355362
18: 1.8319216, 23.2061729, 1.8202538, 23.2397537, -18.7726974, 18.7419319
19: -0.8213892, 11.5533085, -0.8289261, 11.5566244, -11.0328941, 11.0362415
20: -4.4432716, 9.5763245, -4.4553137, 9.5863104, -13.3243103, 13.3263474
21: -1.3717422, 15.5818005, -1.3831649, 15.5945673, -15.9861069, 15.9842987
22: -3.0776050, 11.4769688, -3.0857060, 11.4841433, -13.5486565, 13.5433731
23: -1.3567805, 15.6361828, -1.3628883, 15.6442699, -13.2432976, 13.2284470
24: -1.8840332, 16.2716293, -1.8910732, 16.2899857, -15.0444221, 15.0216827
25: -2.6999431, 16.4039268, -2.7100377, 16.4119835, -17.5163727, 17.5130157
26: -5.3739738, 21.1046257, -5.3870845, 21.1403236, -25.3643188, 25.3354568
27: -0.4238358, 15.5633917, -0.4358692, 15.5883904, -13.4585915, 13.4426575
28: -1.4757776, 15.4893045, -1.4831524, 15.4983654, -14.1415863, 14.1273384
29: -2.0500174, 12.6472187, -2.0543838, 12.6651297, -11.2464485, 11.2155724
30: -8.1377573, 14.7652950, -8.1486549, 14.7940388, -20.0913162, 20.0534363
31: 0.5362120, 16.0149498, 0.5256543, 16.0209732, -14.2785187, 14.2797318
32: -21.9895401, 2.0032697, -22.0030937, 2.0114284, -18.9075623, 18.9140167
33: -39.6697197, -10.4769039, -39.7008514, -10.4652824, -20.9022522, 20.9380646
34: -33.3815193, -10.0917912, -33.4020767, -10.0835295, -17.4627686, 17.4592209
35: -24.0556297, -0.8246055, -24.0725803, -0.8162105, -18.7436066, 18.7523041
36: -20.7910881, 5.2755818, -20.8003273, 5.2824454, -20.0752335, 20.0816650
37: -32.2764359, -2.7063551, -32.2938919, -2.6937919, -25.9368134, 25.9332199
38: -28.7486858, 0.6113558, -28.7619324, 0.6241536, -24.3833923, 24.3793716
39: -43.9204025, -10.2629471, -43.9530983, -10.2568378, -23.0299454, 23.0846100
40: -31.3212299, -13.0401020, -31.3357124, -13.0345478, -12.8971252, 12.9102097
41: -19.8705845, 2.0357602, -19.8817501, 2.0451100, -18.7482681, 18.7499466
42: -20.0844288, -3.5486891, -20.0932751, -3.5399423, -13.5556831, 13.5459099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=146, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4589179
time: 23.26 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4717248
time: 18.90 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -29.3205509, -2.3194776, -29.3434143, -2.3821745, -18.0984421, 18.2175140
1: -13.7389936, 2.7348461, -13.7430849, 2.6995687, -11.8667297, 11.9218788
2: -12.0191193, 4.1260324, -12.0273285, 4.0838437, -10.5899315, 10.6535625
3: -21.0934296, -0.8015361, -21.1174965, -0.8655562, -16.1912460, 16.2958527
4: -19.4321976, 2.7615972, -19.4460125, 2.7084045, -14.4347382, 14.5279922
5: -15.4993629, 4.3023801, -15.5216751, 4.2441692, -15.3405609, 15.4371109
6: -21.6998882, -0.4507885, -21.6823921, -0.4459114, -16.7427063, 16.7070885
7: -18.7228718, 2.4849100, -18.7364655, 2.4445391, -16.9431992, 17.0143890
8: -28.8716278, -1.3337593, -28.8795853, -1.3812351, -17.8598175, 17.9298325
9: -19.0954399, 2.5585589, -19.1170063, 2.4952950, -17.8134308, 17.9264603
10: -16.8849430, 5.3912606, -16.8596039, 5.3715796, -19.8951302, 19.8771553
11: -2.7730260, 15.7798052, -2.7169034, 15.8004131, -17.1924667, 17.0932503
12: -17.4381790, 13.0874138, -17.3748283, 13.1217518, -24.3815231, 24.2779083
13: -30.3888893, -1.5472307, -30.4021893, -1.5946760, -20.8755569, 20.9684944
14: -34.0970497, 0.2636576, -34.0844574, 0.2796917, -29.3027954, 29.2594986
15: -15.3222141, 5.2180719, -15.3384705, 5.1836929, -18.5870667, 18.6382027
16: -15.4912567, 6.3565788, -15.4899855, 6.3040476, -19.2508507, 19.3166199
17: -23.0994759, 1.7165160, -23.0712242, 1.7506919, -23.2361450, 23.1586838
18: 1.7474866, 23.2320499, 1.8176455, 23.2486019, -18.8665771, 18.7657852
19: -0.8619304, 11.5582123, -0.8303380, 11.5568924, -11.0713577, 11.0479069
20: -4.4885035, 9.5811615, -4.4581270, 9.5864878, -13.3685608, 13.3390121
21: -1.4078474, 15.5895729, -1.3856587, 15.5951691, -16.0213394, 16.0001450
22: -3.1057713, 11.4803333, -3.0869274, 11.4854679, -13.5789375, 13.5548782
23: -1.4107985, 15.6535263, -1.3643961, 15.6527128, -13.3008385, 13.2408562
24: -1.9425039, 16.2943363, -1.8919182, 16.2986584, -15.1092987, 15.0401649
25: -2.7509203, 16.4122887, -2.7115831, 16.4139328, -17.5653152, 17.5238800
26: -5.4666309, 21.1187038, -5.3891521, 21.1421700, -25.4631958, 25.3536224
27: -0.4722104, 15.5780602, -0.4375801, 15.5942659, -13.5107880, 13.4550018
28: -1.5208154, 15.4984741, -1.4849234, 15.5009918, -14.1860733, 14.1412163
29: -2.0828912, 12.6653767, -2.0549064, 12.6732388, -11.2888718, 11.2299805
30: -8.1860113, 14.7848883, -8.1503773, 14.8025408, -20.1445465, 20.0715027
31: 0.4858327, 16.0201607, 0.5234075, 16.0220184, -14.3287544, 14.2946053
32: -22.0070267, 2.0067763, -22.0030518, 2.0118828, -18.9311142, 18.9190216
33: -39.6875458, -10.4308138, -39.6972771, -10.4634895, -20.9294472, 20.9787178
34: -33.4144096, -10.0836411, -33.4035225, -10.0821552, -17.5025597, 17.4714890
35: -24.0673332, -0.7986479, -24.0705833, -0.8143024, -18.7631760, 18.7771263
36: -20.8087502, 5.2798452, -20.8015385, 5.2828741, -20.0994949, 20.0862274
37: -32.3132019, -2.7082214, -32.2962418, -2.6977434, -25.9869461, 25.9399796
38: -28.8023396, 0.6222253, -28.7652168, 0.6263986, -24.4577026, 24.3935165
39: -43.9552765, -10.2209244, -43.9603043, -10.2560759, -23.0658340, 23.1269150
40: -31.3418350, -13.0364027, -31.3379650, -13.0343113, -12.9257927, 12.9161339
41: -19.9008141, 2.0479357, -19.8828468, 2.0482693, -18.7860718, 18.7642212
42: -20.1166286, -3.5397592, -20.0945129, -3.5386038, -13.5969124, 13.5560455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=146, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4720691
time: 21.07 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4848774
time: 26.65 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -29.3489075, -2.3048363, -29.3575211, -2.3755455, -18.1409645, 18.2530136
1: -13.7380562, 2.7251086, -13.7417212, 2.7013593, -11.8727188, 11.9117584
2: -12.0251827, 4.1202669, -12.0292635, 4.0867834, -10.6019745, 10.6534233
3: -21.1254959, -0.7838302, -21.1353302, -0.8605185, -16.2244644, 16.3312836
4: -19.4430847, 2.7654076, -19.4522629, 2.7136593, -14.4578934, 14.5452881
5: -15.5322123, 4.3159666, -15.5387430, 4.2498374, -15.3764725, 15.4680099
6: -21.7069340, -0.4366455, -21.6874390, -0.4438467, -16.7839890, 16.7340813
7: -18.7367020, 2.4690986, -18.7412663, 2.4494562, -16.9633331, 17.0050507
8: -28.8722229, -1.3716965, -28.8786392, -1.3799534, -17.8687057, 17.8931961
9: -19.1228275, 2.5843046, -19.1316357, 2.5020804, -17.8500290, 17.9642563
10: -16.8859253, 5.4216747, -16.8699169, 5.3780189, -19.8981667, 19.9256783
11: -2.8381722, 15.8080702, -2.7282431, 15.8187027, -17.2726822, 17.1328621
12: -17.4698868, 13.1593952, -17.3824234, 13.1572447, -24.4480209, 24.3547974
13: -30.3978405, -1.5193973, -30.4049206, -1.5846334, -20.8993835, 20.9909973
14: -34.2257881, 0.3321705, -34.0995598, 0.3252401, -29.4694519, 29.3300705
15: -15.3557043, 5.2510166, -15.3565063, 5.1907859, -18.6175079, 18.6768990
16: -15.5020227, 6.3372278, -15.4901714, 6.3085194, -19.2779961, 19.3070679
17: -23.1938400, 1.8101304, -23.0778770, 1.8096068, -23.3898239, 23.2566986
18: 1.7256174, 23.2682018, 1.8116460, 23.2709484, -18.9019089, 18.7989616
19: -0.8857899, 11.5602999, -0.8363161, 11.5596161, -11.0935020, 11.0722084
20: -4.5339823, 9.6000252, -4.4668288, 9.5973473, -13.4220695, 13.3552971
21: -1.4740276, 15.6121483, -1.3944850, 15.6102829, -16.0983658, 16.0150261
22: -3.1082635, 11.4947224, -3.0930505, 11.4907932, -13.5856743, 13.5715027
23: -1.4162951, 15.6403198, -1.3686409, 15.6458864, -13.3063622, 13.2484207
24: -1.9469395, 16.2937813, -1.8970146, 16.3005829, -15.1148071, 15.0450859
25: -2.7521615, 16.4141979, -2.7189546, 16.4172058, -17.5747604, 17.5328484
26: -5.5279760, 21.1767921, -5.3983474, 21.1757393, -25.5551987, 25.4105148
27: -0.4999928, 15.6012478, -0.4467330, 15.6063299, -13.5466919, 13.4850006
28: -1.5423861, 15.5030041, -1.4904847, 15.5050964, -14.2107506, 14.1586761
29: -2.0969505, 12.6701927, -2.0592623, 12.6757650, -11.3093338, 11.2411919
30: -8.2431126, 14.8179398, -8.1597815, 14.8225708, -20.2241135, 20.1123734
31: 0.4646645, 16.0289593, 0.5151343, 16.0269852, -14.3382111, 14.3217506
32: -22.0190887, 2.0571785, -22.0168648, 2.0187936, -18.9374390, 18.9764938
33: -39.7424316, -10.3647614, -39.7336388, -10.4546318, -20.9729309, 21.0816269
34: -33.4300499, -10.0194359, -33.4257622, -10.0770960, -17.5056076, 17.5591965
35: -24.0914974, -0.7839851, -24.0858059, -0.8101242, -18.7865982, 18.8042259
36: -20.8128662, 5.3037777, -20.8067780, 5.2895117, -20.1198349, 20.1104736
37: -32.3277168, -2.6704164, -32.3096008, -2.6822300, -26.0521088, 25.9708328
38: -28.7866459, 0.6498208, -28.7724152, 0.6328459, -24.4701309, 24.4288712
39: -43.9835091, -10.1548891, -43.9825592, -10.2491951, -23.0882645, 23.2150993
40: -31.3514500, -13.0176878, -31.3490620, -13.0283175, -12.9611740, 12.9436264
41: -19.8959312, 2.0766842, -19.8930607, 2.0523326, -18.7847061, 18.8015442
42: -20.1126480, -3.5131137, -20.1033230, -3.5334430, -13.6280327, 13.5835304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=146, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4822953
time: 21.01 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4950896
time: 23.38 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -29.3773613, -2.2339678, -29.3718719, -2.3742323, -18.1637268, 18.3395500
1: -13.7499580, 2.7642517, -13.7475100, 2.7023365, -11.8820877, 11.9573669
2: -12.0400772, 4.1670213, -12.0369177, 4.0876126, -10.6149521, 10.7081280
3: -21.1513710, -0.7105904, -21.1483002, -0.8590612, -16.2460098, 16.4194298
4: -19.4723473, 2.8250642, -19.4662056, 2.7147427, -14.4806366, 14.6174049
5: -15.5546942, 4.3818121, -15.5498886, 4.2509389, -15.3955193, 15.5461464
6: -21.7316723, -0.4262123, -21.6886425, -0.4410572, -16.8156967, 16.7446327
7: -18.7598381, 2.5153890, -18.7529087, 2.4501410, -16.9831390, 17.0632553
8: -28.8988876, -1.3207874, -28.8911724, -1.3789692, -17.8891373, 17.9569817
9: -19.1593380, 2.6572042, -19.1486816, 2.5032401, -17.8812561, 18.0542717
10: -16.9266033, 5.4517150, -16.8719635, 5.3792567, -19.9460106, 19.9552689
11: -2.9081378, 15.8409214, -2.7298474, 15.8339348, -17.3534470, 17.1589050
12: -17.5447388, 13.1889133, -17.3845749, 13.1697788, -24.5357132, 24.3858871
13: -30.4214497, -1.4580741, -30.4150009, -1.5824475, -20.9186172, 21.0637512
14: -34.2666626, 0.3532071, -34.1048203, 0.3261333, -29.5208740, 29.3556747
15: -15.3714085, 5.2967877, -15.3615980, 5.1928949, -18.6350861, 18.7274437
16: -15.5321779, 6.3956518, -15.5003986, 6.3091803, -19.3047485, 19.3755722
17: -23.2341728, 1.8324025, -23.0796738, 1.8122444, -23.4349136, 23.2798767
18: 1.6411710, 23.2940769, 1.8090043, 23.2798347, -18.9957886, 18.8228226
19: -0.9263735, 11.5652094, -0.8377361, 11.5598688, -11.1319733, 11.0838718
20: -4.5792542, 9.6048679, -4.4696236, 9.5975189, -13.4663925, 13.3679657
21: -1.5101995, 15.6199131, -1.3969860, 15.6108809, -16.1336212, 16.0309029
22: -3.1364202, 11.4981213, -3.0942585, 11.4920969, -13.6159897, 13.5830193
23: -1.4703398, 15.6576939, -1.3701458, 15.6543121, -13.3639641, 13.2608452
24: -2.0054193, 16.3165054, -1.8978534, 16.3092194, -15.1797028, 15.0635681
25: -2.8031874, 16.4225960, -2.7205114, 16.4191608, -17.6237488, 17.5437126
26: -5.6206622, 21.1908360, -5.4005060, 21.1775742, -25.6540604, 25.4287109
27: -0.5483150, 15.6159067, -0.4484634, 15.6122065, -13.5988770, 13.4973297
28: -1.5874472, 15.5121746, -1.4922538, 15.5077333, -14.2553062, 14.1725540
29: -2.1298237, 12.6883602, -2.0597534, 12.6838675, -11.3517723, 11.2556190
30: -8.2913418, 14.8374662, -8.1615238, 14.8310184, -20.2773666, 20.1304703
31: 0.4142036, 16.0341797, 0.5128641, 16.0280418, -14.3884850, 14.3366508
32: -22.0365677, 2.0606971, -22.0168419, 2.0192680, -18.9609604, 18.9814568
33: -39.7602234, -10.3186722, -39.7301064, -10.4528618, -21.0000229, 21.1222839
34: -33.4628754, -10.0113220, -33.4272079, -10.0757189, -17.5453568, 17.5714912
35: -24.1032314, -0.7580655, -24.0838528, -0.8082070, -18.8062134, 18.8290634
36: -20.8304806, 5.3080330, -20.8080120, 5.2898993, -20.1440811, 20.1149979
37: -32.3644943, -2.6722693, -32.3118973, -2.6861610, -26.1023026, 25.9775696
38: -28.8403034, 0.6607389, -28.7756958, 0.6351233, -24.5444412, 24.4429932
39: -44.0184097, -10.1129723, -43.9897652, -10.2484503, -23.1241074, 23.2574120
40: -31.3720093, -13.0139704, -31.3513107, -13.0280638, -12.9897995, 12.9495468
41: -19.9261227, 2.0888772, -19.8941803, 2.0554912, -18.8224792, 18.8158417
42: -20.1448040, -3.5041952, -20.1045532, -3.5321133, -13.6692429, 13.5936699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=146, inp2_unstable=147, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4954366
time: 16.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5082337, upper bound: 10.5082330
time: 24.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 43.46 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 43.46
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4589179
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 43.46
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4717248
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 43.46
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4720691
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 43.46
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4848774
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 43.46
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4822953
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 43.46
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4950896
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 43.46
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4954366
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 43.46
Output dim: 18, lower bound: -10.5082337, upper bound: 10.5082330

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -29.3485050, -2.3183022, -29.3567467, -2.4013538, -18.1171570, 18.2400246
1: -13.7375660, 2.7167954, -13.7407856, 2.6855044, -11.8577499, 11.9031258
2: -12.0251217, 4.1105204, -12.0291166, 4.0682068, -10.5841980, 10.6437855
3: -21.1253204, -0.7935138, -21.1350231, -0.8790488, -16.2056656, 16.3211060
4: -19.4424133, 2.7519040, -19.4509544, 2.6879272, -14.4335480, 14.5312691
5: -15.5319710, 4.3045778, -15.5383348, 4.2280049, -15.3543091, 15.4560852
6: -21.7065296, -0.4378881, -21.6865959, -0.4462070, -16.7805710, 16.7312813
7: -18.7361908, 2.4577885, -18.7402687, 2.4279089, -16.9409790, 16.9925766
8: -28.8716412, -1.3883061, -28.8775101, -1.4117737, -17.8384018, 17.8761444
9: -19.1221333, 2.5735471, -19.1303368, 2.4815733, -17.8295898, 17.9522781
10: -16.8844090, 5.4159007, -16.8670311, 5.3670182, -19.8853378, 19.9167442
11: -2.8256793, 15.8076572, -2.7043762, 15.8178711, -17.2607803, 17.1114616
12: -17.4620247, 13.1579666, -17.3674698, 13.1545229, -24.4373779, 24.3394089
13: -30.3964996, -1.5218601, -30.4024010, -1.5893955, -20.8929825, 20.9854317
14: -34.2219543, 0.3301888, -34.0922546, 0.3214092, -29.4586487, 29.3175507
15: -15.3546925, 5.2437086, -15.3545580, 5.1769381, -18.6039276, 18.6679115
16: -15.5004816, 6.3349977, -15.4872065, 6.3042717, -19.2713318, 19.3009186
17: -23.1891479, 1.8090022, -23.0688972, 1.8074868, -23.3798141, 23.2411575
18: 1.7409611, 23.2678108, 1.8409295, 23.2702370, -18.8858490, 18.7692032
19: -0.8806467, 11.5601902, -0.8264923, 11.5593967, -11.0880775, 11.0623589
20: -4.5306416, 9.5989246, -4.4604130, 9.5952368, -13.4161987, 13.3474903
21: -1.4666643, 15.6118002, -1.3804479, 15.6096201, -16.0913773, 16.0032997
22: -3.1054463, 11.4942408, -3.0876913, 11.4898586, -13.5816803, 13.5656815
23: -1.4044762, 15.6400194, -1.3460684, 15.6453133, -13.2939110, 13.2256927
24: -1.9337320, 16.2936230, -1.8717966, 16.3002796, -15.1011200, 15.0192947
25: -2.7413969, 16.4139404, -2.6983562, 16.4167290, -17.5636902, 17.5124283
26: -5.5115256, 21.1763783, -5.3669405, 21.1749649, -25.5379105, 25.3785324
27: -0.4972610, 15.6006994, -0.4415159, 15.6053219, -13.5427933, 13.4791260
28: -1.5330529, 15.5024681, -1.4726295, 15.5040894, -14.2003365, 14.1404800
29: -2.0920644, 12.6698236, -2.0499008, 12.6751089, -11.3038025, 11.2321053
30: -8.2281723, 14.8175745, -8.1312580, 14.8218212, -20.2093277, 20.0855713
31: 0.4681950, 16.0285454, 0.5218873, 16.0262222, -14.3325424, 14.3146820
32: -22.0183659, 2.0498505, -22.0155144, 2.0049911, -18.9191971, 18.9657440
33: -39.7377472, -10.3656864, -39.7247849, -10.4563866, -20.9662170, 21.0716553
34: -33.4206924, -10.0203056, -33.4079247, -10.0787516, -17.4942017, 17.5401535
35: -24.0801582, -0.7850170, -24.0641441, -0.8120208, -18.7728500, 18.7809258
36: -20.8062859, 5.3033690, -20.7942238, 5.2886925, -20.1124496, 20.0974426
37: -32.3135643, -2.6708241, -32.2825470, -2.6829014, -26.0369263, 25.9428101
38: -28.7816086, 0.6486850, -28.7628651, 0.6306920, -24.4629364, 24.4177094
39: -43.9815254, -10.1555176, -43.9787598, -10.2503376, -23.0844040, 23.2095947
40: -31.3507271, -13.0180235, -31.3477020, -13.0289316, -12.9585419, 12.9398537
41: -19.8943768, 2.0749688, -19.8901196, 2.0490742, -18.7798309, 18.7969208
42: -20.1117020, -3.5163960, -20.1015186, -3.5397258, -13.6194344, 13.5776443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=146, inp2_unstable=146, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4810190
time: 20.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4810190
time: 21.11 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -29.3486977, -2.3075762, -29.4125690, -2.3697910, -18.1387405, 18.3033409
1: -13.7379608, 2.7234266, -13.7767992, 2.7046161, -11.8716049, 11.9447174
2: -12.0251179, 4.1183195, -12.0690174, 4.0901880, -10.5994987, 10.6906548
3: -21.1254349, -0.7858100, -21.1768360, -0.8526020, -16.2279282, 16.3707695
4: -19.4429436, 2.7626944, -19.5228462, 2.7173910, -14.4535980, 14.6138535
5: -15.5320711, 4.3137183, -15.5849285, 4.2560081, -15.3776512, 15.5120010
6: -21.7067547, -0.4369946, -21.6974297, -0.4324274, -16.7957993, 16.7441025
7: -18.7364941, 2.4669375, -18.7926903, 2.4528561, -16.9607620, 17.0548630
8: -28.8719902, -1.3750129, -28.9563599, -1.3778811, -17.8608932, 17.9671555
9: -19.1226578, 2.5821006, -19.1807060, 2.5085709, -17.8495636, 18.0107803
10: -16.8854942, 5.4203663, -16.9013939, 5.3882694, -19.9073563, 19.9577789
11: -2.8354442, 15.8078756, -2.7391021, 15.8794756, -17.3292160, 17.1393127
12: -17.4683800, 13.1590338, -17.3874149, 13.2066879, -24.4983139, 24.3558731
13: -30.3975525, -1.5201182, -30.4206314, -1.5638556, -20.9311981, 20.9995575
14: -34.2248306, 0.3313599, -34.1294098, 0.3373432, -29.4944305, 29.3452225
15: -15.3553858, 5.2492847, -15.3992386, 5.1997962, -18.6231079, 18.7160568
16: -15.5015717, 6.3366852, -15.5137939, 6.3172779, -19.3037949, 19.3178635
17: -23.1914196, 1.8097193, -23.0881329, 1.8571301, -23.4509201, 23.2537994
18: 1.7287393, 23.2680969, 1.8001432, 23.3285179, -18.9566650, 18.8063774
19: -0.8845840, 11.5602589, -0.8473272, 11.5744038, -11.1073685, 11.0827751
20: -4.5330997, 9.5991259, -4.4798193, 9.6023827, -13.4263153, 13.3680649
21: -1.4723334, 15.6120205, -1.4056010, 15.6316328, -16.1151657, 16.0257034
22: -3.1072257, 11.4945354, -3.1010678, 11.4984303, -13.5893059, 13.5855713
23: -1.4138436, 15.6402512, -1.3749638, 15.6901636, -13.3483429, 13.2507362
24: -1.9442849, 16.2937241, -1.9087667, 16.3543777, -15.1661873, 15.0535049
25: -2.7500687, 16.4140587, -2.7321396, 16.4626961, -17.6181870, 17.5443802
26: -5.5246916, 21.1766300, -5.4116592, 21.2262077, -25.6031723, 25.4233856
27: -0.4992270, 15.6011028, -0.4595737, 15.6209831, -13.5602264, 13.4957390
28: -1.5403829, 15.5028191, -1.4989181, 15.5441036, -14.2474747, 14.1650276
29: -2.0960245, 12.6700745, -2.0630646, 12.6937132, -11.3281860, 11.2421608
30: -8.2399902, 14.8177261, -8.1711121, 14.8960199, -20.2936401, 20.1176147
31: 0.4658055, 16.0288372, 0.5048895, 16.0327282, -14.3342743, 14.3370361
32: -22.0188560, 2.0556922, -22.0576611, 2.0326958, -18.9426956, 19.0261002
33: -39.7412796, -10.3650885, -39.7447128, -10.4369593, -20.9871140, 21.0912285
34: -33.4281693, -10.0196886, -33.4358826, -10.0256624, -17.5551491, 17.5625992
35: -24.0892525, -0.7843087, -24.0971546, -0.7536764, -18.8403397, 18.8093185
36: -20.8115234, 5.3036270, -20.8141136, 5.3259144, -20.1563873, 20.1160126
37: -32.3247452, -2.6705451, -32.3249779, -2.6163459, -26.1150970, 25.9795380
38: -28.7855892, 0.6495600, -28.7869682, 0.6636076, -24.5048904, 24.4421005
39: -43.9830360, -10.1551657, -44.0058975, -10.2430820, -23.1077042, 23.2271271
40: -31.3512115, -13.0177994, -31.3616638, -13.0247459, -12.9749718, 12.9436874
41: -19.8954048, 2.0763040, -19.9009399, 2.0708990, -18.8038177, 18.8136520
42: -20.1123466, -3.5140188, -20.1196747, -3.5237639, -13.6347618, 13.6034622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=146, inp2_unstable=146, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4938150
time: 22.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4938150
time: 25.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -29.3769493, -2.2474499, -29.3710670, -2.4000578, -18.1399231, 18.3265228
1: -13.7494745, 2.7559311, -13.7465630, 2.6864572, -11.8671341, 11.9487114
2: -12.0400238, 4.1572676, -12.0367851, 4.0690103, -10.5971718, 10.6984997
3: -21.1511860, -0.7202649, -21.1479778, -0.8775663, -16.2272110, 16.4092903
4: -19.4716644, 2.8115406, -19.4649582, 2.6889620, -14.4563026, 14.6034012
5: -15.5544462, 4.3704004, -15.5494499, 4.2291346, -15.3733406, 15.5342445
6: -21.7312527, -0.4275017, -21.6878147, -0.4434252, -16.8122864, 16.7418175
7: -18.7593365, 2.5040917, -18.7518978, 2.4285684, -16.9608002, 17.0508041
8: -28.8983040, -1.3374157, -28.8900604, -1.4107475, -17.8588486, 17.9399376
9: -19.1586418, 2.6464787, -19.1473465, 2.4826915, -17.8607788, 18.0423126
10: -16.9251022, 5.4458971, -16.8690681, 5.3682618, -19.9331894, 19.9462967
11: -2.8956788, 15.8404617, -2.7060189, 15.8331070, -17.3415604, 17.1374893
12: -17.5368710, 13.1874790, -17.3695793, 13.1670094, -24.5250549, 24.3704529
13: -30.4201050, -1.4605494, -30.4124336, -1.5872169, -20.9122467, 21.0581894
14: -34.2628326, 0.3512130, -34.0974655, 0.3223295, -29.5100632, 29.3431702
15: -15.3703785, 5.2894650, -15.3596287, 5.1790237, -18.6214828, 18.7184868
16: -15.5306339, 6.3934193, -15.4974384, 6.3048916, -19.2981262, 19.3693771
17: -23.2294521, 1.8312473, -23.0706673, 1.8101156, -23.4248657, 23.2643433
18: 1.6565280, 23.2936993, 1.8382936, 23.2791023, -18.9797363, 18.7930183
19: -0.9212270, 11.5651035, -0.8279142, 11.5596533, -11.1265755, 11.0740051
20: -4.5759029, 9.6037626, -4.4631968, 9.5953960, -13.4604950, 13.3601627
21: -1.5028248, 15.6195660, -1.3829446, 15.6102257, -16.1266479, 16.0191574
22: -3.1335902, 11.4976406, -3.0888982, 11.4911575, -13.6119995, 13.5771980
23: -1.4585419, 15.6574173, -1.3476038, 15.6537542, -13.3515167, 13.2381191
24: -1.9922090, 16.3163242, -1.8726373, 16.3089371, -15.1660271, 15.0377655
25: -2.7924004, 16.4223480, -2.6999321, 16.4186821, -17.6126404, 17.5233154
26: -5.6042013, 21.1904411, -5.3690534, 21.1768036, -25.6367645, 25.3967285
27: -0.5456095, 15.6153927, -0.4432497, 15.6112108, -13.5949860, 13.4914742
28: -1.5780964, 15.5116262, -1.4744129, 15.5067329, -14.2448921, 14.1543083
29: -2.1249223, 12.6879940, -2.0503833, 12.6831818, -11.3462486, 11.2465477
30: -8.2764482, 14.8371172, -8.1329451, 14.8302460, -20.2625427, 20.1036606
31: 0.4177756, 16.0337734, 0.5196028, 16.0272598, -14.3828125, 14.3295708
32: -22.0358391, 2.0534182, -22.0154896, 2.0054960, -18.9427185, 18.9707184
33: -39.7555046, -10.3196669, -39.7212753, -10.4546528, -20.9933434, 21.1123505
34: -33.4535675, -10.0121737, -33.4093590, -10.0773382, -17.5339355, 17.5524597
35: -24.0918598, -0.7590342, -24.0621529, -0.8101165, -18.7924347, 18.8057976
36: -20.8239250, 5.3076191, -20.7954216, 5.2890983, -20.1367340, 20.1019745
37: -32.3503494, -2.6726365, -32.2848282, -2.6868439, -26.0871353, 25.9495010
38: -28.8352909, 0.6595907, -28.7661095, 0.6329479, -24.5372314, 24.4318695
39: -44.0164070, -10.1135387, -43.9859695, -10.2495346, -23.1202621, 23.2518845
40: -31.3713284, -13.0142822, -31.3499393, -13.0286884, -12.9872055, 12.9457741
41: -19.9246101, 2.0871656, -19.8912010, 2.0522346, -18.8176193, 18.8112259
42: -20.1438828, -3.5074873, -20.1027470, -3.5383844, -13.6606598, 13.5877914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=146, inp2_unstable=146, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4941786
time: 26.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4941786
time: 18.08 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -29.3771362, -2.2367177, -29.4269009, -2.3684902, -18.1615067, 18.3898621
1: -13.7498398, 2.7625608, -13.7825642, 2.7055726, -11.8809853, 11.9903183
2: -12.0400209, 4.1650896, -12.0766630, 4.0910473, -10.6124763, 10.7453518
3: -21.1513042, -0.7125626, -21.1898098, -0.8511415, -16.2494469, 16.4589424
4: -19.4721851, 2.8223596, -19.5367966, 2.7183933, -14.4763489, 14.6859512
5: -15.5545607, 4.3795576, -15.5960350, 4.2570672, -15.3966713, 15.5901260
6: -21.7314987, -0.4265442, -21.6986637, -0.4296780, -16.8275223, 16.7546616
7: -18.7596359, 2.5132465, -18.8043041, 2.4535542, -16.9806061, 17.1130829
8: -28.8986702, -1.3240991, -28.9689407, -1.3769112, -17.8813171, 18.0309601
9: -19.1591797, 2.6549807, -19.1977158, 2.5097158, -17.8807678, 18.1007996
10: -16.9262104, 5.4503798, -16.9034348, 5.3894854, -19.9552002, 19.9873695
11: -2.9054375, 15.8406954, -2.7407117, 15.8947086, -17.4100113, 17.1653595
12: -17.5432568, 13.1885538, -17.3895321, 13.2192087, -24.5860062, 24.3869400
13: -30.4211178, -1.4587798, -30.4306755, -1.5616879, -20.9504242, 21.0723381
14: -34.2656784, 0.3524055, -34.1345978, 0.3382006, -29.5458527, 29.3708267
15: -15.3710632, 5.2950678, -15.4042854, 5.2019172, -18.6406784, 18.7665787
16: -15.5317354, 6.3950872, -15.5240078, 6.3178768, -19.3305702, 19.3863411
17: -23.2317314, 1.8319535, -23.0899143, 1.8597374, -23.4960251, 23.2769699
18: 1.6443143, 23.2939816, 1.7975154, 23.3373909, -19.0505066, 18.8302116
19: -0.9251513, 11.5651684, -0.8487425, 11.5746689, -11.1458473, 11.0944424
20: -4.5783610, 9.6039734, -4.4826279, 9.6025562, -13.4706078, 13.3807297
21: -1.5085063, 15.6197910, -1.4080925, 15.6322212, -16.1504135, 16.0415611
22: -3.1353941, 11.4979362, -3.1022730, 11.4997520, -13.6195946, 13.5970917
23: -1.4678822, 15.6576405, -1.3764763, 15.6985989, -13.4059219, 13.2631721
24: -2.0027485, 16.3164330, -1.9095926, 16.3630447, -15.2310791, 15.0719719
25: -2.8010821, 16.4224606, -2.7336907, 16.4647064, -17.6671829, 17.5552826
26: -5.6173820, 21.1906624, -5.4137897, 21.2280121, -25.7020264, 25.4415741
27: -0.5475435, 15.6157627, -0.4612713, 15.6268616, -13.6123962, 13.5080872
28: -1.5854487, 15.5119743, -1.5006919, 15.5467424, -14.2920036, 14.1788712
29: -2.1288667, 12.6882353, -2.0635540, 12.7018299, -11.3706207, 11.2565918
30: -8.2882509, 14.8372507, -8.1727867, 14.9044762, -20.3468933, 20.1356735
31: 0.4153528, 16.0340500, 0.5026474, 16.0337830, -14.3845520, 14.3519135
32: -22.0363293, 2.0592313, -22.0576038, 2.0331483, -18.9662476, 19.0310707
33: -39.7590561, -10.3189821, -39.7411423, -10.4351921, -21.0142212, 21.1318817
34: -33.4610443, -10.0115576, -33.4373169, -10.0243092, -17.5949059, 17.5748634
35: -24.1009312, -0.7583647, -24.0951538, -0.7517662, -18.8599091, 18.8341599
36: -20.8292217, 5.3078518, -20.8153648, 5.3263025, -20.1806107, 20.1205521
37: -32.3615608, -2.6724253, -32.3272247, -2.6202955, -26.1652374, 25.9862442
38: -28.8392735, 0.6604319, -28.7902451, 0.6658812, -24.5791779, 24.4562149
39: -44.0179138, -10.1132183, -44.0131378, -10.2422981, -23.1435471, 23.2694168
40: -31.3717861, -13.0140829, -31.3639050, -13.0245094, -13.0036011, 12.9496078
41: -19.9255905, 2.0885050, -19.9020710, 2.0740705, -18.8415833, 18.8279190
42: -20.1445274, -3.5051088, -20.1208687, -3.5224013, -13.6759682, 13.6135902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=146, inp2_unstable=146, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=162, inp2_unstable=162, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=12, inp2_unstable=12, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=29, inp2_unstable=29, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1006

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.4891339, upper bound: 10.5069726
time: 24.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -10.5069733, upper bound: 10.5069726
time: 19.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 45.92 seconds
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 45.92
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4810190
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 45.92
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4810190
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 45.92
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4938150
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 45.92
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4938150
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 45.92
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4941786
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 45.92
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4941786
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 45.92
Output dim: 18, lower bound: -10.4891339, upper bound: 10.5069726
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 45.92
Output dim: 18, lower bound: -10.5069733, upper bound: 10.5069726

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 33.15 + 733.39 = 766.53 seconds
