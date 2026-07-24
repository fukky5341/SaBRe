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
execution time: IAR + RelationalAnalysis = 2.77 + 31.62 = 34.39 seconds
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
time: 29.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5160992, upper bound: 10.5160989
time: 22.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 52.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 52.94
Output dim: 18, lower bound: -10.5140667, upper bound: 10.4928949
IS_A2, status: Status.UNKNOWN, split count: 1, time: 52.94
Output dim: 18, lower bound: -10.5160992, upper bound: 10.5160989

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5079954, upper bound: 10.4650733
time: 27.25 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5128896, upper bound: 10.4917163
time: 18.70 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1629

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5100558, upper bound: 10.4882992
time: 26.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5149297, upper bound: 10.5149292
time: 27.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 56.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 56.37
Output dim: 18, lower bound: -10.5079954, upper bound: 10.4650733
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 56.37
Output dim: 18, lower bound: -10.5128896, upper bound: 10.4917163
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 56.37
Output dim: 18, lower bound: -10.5100558, upper bound: 10.4882992
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 56.37
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

Time for backsubstitution: 2.16 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5039856, upper bound: 10.4493261
time: 21.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5039856, upper bound: 10.4611604
time: 25.07 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5096145, upper bound: 10.4752886
time: 22.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5096145, upper bound: 10.4884318
time: 24.32 seconds

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

Time for backsubstitution: 2.18 seconds

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
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5060186, upper bound: 10.4493261
time: 47.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5060186, upper bound: 10.4843029
time: 22.36 seconds

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5117014, upper bound: 10.4985757
time: 19.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5117014, upper bound: 10.5117008
time: 18.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 40.75 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 40.75
Output dim: 18, lower bound: -10.5039856, upper bound: 10.4493261
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 40.75
Output dim: 18, lower bound: -10.5039856, upper bound: 10.4611604
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 40.75
Output dim: 18, lower bound: -10.5096145, upper bound: 10.4752886
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 40.75
Output dim: 18, lower bound: -10.5096145, upper bound: 10.4884318
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 40.75
Output dim: 18, lower bound: -10.5060186, upper bound: 10.4493261
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 40.75
Output dim: 18, lower bound: -10.5060186, upper bound: 10.4843029
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 40.75
Output dim: 18, lower bound: -10.5117014, upper bound: 10.4985757
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 40.75
Output dim: 18, lower bound: -10.5117014, upper bound: 10.5117008

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -29.2873878, -2.4451513, -29.2979164, -2.4824958, -17.9924507, 18.0429688
1: -13.7233582, 2.6624365, -13.7210846, 2.6375797, -11.8030624, 11.8276825
2: -12.0020809, 4.0479832, -12.0083294, 4.0256715, -10.5241051, 10.5548344
3: -21.0641594, -0.9151030, -21.0913010, -0.9413042, -16.0972443, 16.1523209
4: -19.3982048, 2.6601357, -19.4037590, 2.6318040, -14.3463936, 14.3812523
5: -15.4734831, 4.1942635, -15.4938135, 4.1661530, -15.2498550, 15.3002014
6: -21.6636333, -0.4717512, -21.6578064, -0.4692554, -16.6747513, 16.6613846
7: -18.6965675, 2.4067283, -18.7063789, 2.3855195, -16.8733978, 16.9084930
8: -28.8375816, -1.4102445, -28.8450756, -1.4302378, -17.7947617, 17.8230515
9: -19.0529747, 2.4320045, -19.0664845, 2.3989484, -17.6976395, 17.7445526
10: -16.7989750, 5.3457317, -16.7770119, 5.3396144, -19.7690811, 19.7515945
11: -2.6550860, 15.7415028, -2.6282732, 15.7583237, -17.0305176, 16.9853592
12: -17.3309631, 13.0471745, -17.3125114, 13.0862064, -24.2351379, 24.1779251
13: -30.3568192, -1.7037296, -30.3401871, -1.7667079, -20.7027512, 20.7500687
14: -34.0147018, 0.2282777, -34.0005684, 0.2529383, -29.1806793, 29.1383743
15: -15.2952595, 5.1351404, -15.3078566, 5.1137185, -18.4900665, 18.5225639
16: -15.4501925, 6.2529583, -15.4471645, 6.2222719, -19.1436234, 19.1714363
17: -23.0252228, 1.6769636, -23.0071564, 1.7114692, -23.1178436, 23.0589218
18: 1.8963990, 23.1995716, 1.9365854, 23.2137508, -18.6763077, 18.6236420
19: -0.7872820, 11.5508709, -0.7656245, 11.5508070, -10.9861279, 10.9645958
20: -4.3970861, 9.5735168, -4.3695946, 9.5798664, -13.2709503, 13.2372456
21: -1.3315372, 15.5798473, -1.3082943, 15.5900288, -15.9387856, 15.9084549
22: -3.0417092, 11.4737282, -3.0201428, 11.4724417, -13.4986572, 13.4767532
23: -1.3075523, 15.6338406, -1.2735457, 15.6256876, -13.1715698, 13.1462746
24: -1.8340588, 16.2674751, -1.7993145, 16.2684879, -14.9685516, 14.9355965
25: -2.6557317, 16.4005527, -2.6274128, 16.4039040, -17.4583511, 17.4287834
26: -5.3024783, 21.0977898, -5.2559652, 21.1208344, -25.2668304, 25.1983032
27: -0.3743544, 15.5590496, -0.3465252, 15.5664587, -13.3865204, 13.3534012
28: -1.4303737, 15.4863825, -1.3998446, 15.4819374, -14.0769234, 14.0513382
29: -2.0178769, 12.6443615, -1.9965823, 12.6470928, -11.1898804, 11.1659966
30: -8.0860062, 14.7615776, -8.0541162, 14.7665176, -20.0078125, 19.9717865
31: 0.5824885, 16.0125961, 0.6113391, 16.0128822, -14.2199821, 14.1922951
32: -21.9784851, 1.9848514, -21.9802208, 1.9766536, -18.8605347, 18.8673859
33: -39.6491776, -10.5253448, -39.6501541, -10.5516129, -20.8050842, 20.8331604
34: -33.3433418, -10.0989609, -33.3332901, -10.1068068, -17.3979721, 17.3916588
35: -24.0417500, -0.8578107, -24.0430603, -0.8773873, -18.6685562, 18.6848145
36: -20.7797794, 5.2384644, -20.7682381, 5.2157125, -20.0099640, 20.0147781
37: -32.2569809, -2.7466984, -32.2508354, -2.7648664, -25.8531799, 25.8467178
38: -28.7287159, 0.5788369, -28.7192707, 0.5657854, -24.3144913, 24.3066788
39: -43.8976135, -10.3378754, -43.8776779, -10.3890438, -22.9014816, 22.9252663
40: -31.3085327, -13.0619955, -31.3041916, -13.0727425, -12.8563194, 12.8517456
41: -19.8626652, 2.0196278, -19.8657093, 2.0141320, -18.7050095, 18.7110291
42: -20.0637684, -3.5558443, -20.0546188, -3.5555940, -13.5022926, 13.4904747

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 973
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
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1583
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5005572, upper bound: 10.4330251
time: 22.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5005572, upper bound: 10.4458188
time: 27.62 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -29.3159580, -2.3743200, -29.3122540, -2.4811831, -18.0153160, 18.1294556
1: -13.7353258, 2.7015252, -13.7269039, 2.6385906, -11.8124008, 11.8732872
2: -12.0169506, 4.0947256, -12.0159664, 4.0265393, -10.5371094, 10.6095352
3: -21.0900688, -0.8418212, -21.1042519, -0.9397645, -16.1189041, 16.2405281
4: -19.4274845, 2.7198062, -19.4177361, 2.6329117, -14.3691711, 14.4534340
5: -15.4960232, 4.2601252, -15.5049305, 4.1672773, -15.2689552, 15.3783150
6: -21.6884155, -0.4610682, -21.6590595, -0.4662991, -16.7065582, 16.6720047
7: -18.7196865, 2.4530578, -18.7180290, 2.3862696, -16.8927307, 16.9669342
8: -28.8642616, -1.3593798, -28.8576107, -1.4291945, -17.8162003, 17.8867645
9: -19.0895748, 2.5049019, -19.0835533, 2.4000945, -17.7290001, 17.8346252
10: -16.8398075, 5.3756475, -16.7790527, 5.3408222, -19.8170738, 19.7802315
11: -2.7252059, 15.7744894, -2.6300089, 15.7735691, -17.1114273, 17.0116653
12: -17.4058018, 13.0767059, -17.3147030, 13.0987339, -24.3227921, 24.2091064
13: -30.3807335, -1.6424170, -30.3504848, -1.7644944, -20.7222519, 20.8229828
14: -34.0561676, 0.2493324, -34.0058746, 0.2538729, -29.2326202, 29.1638718
15: -15.3110743, 5.1808629, -15.3130283, 5.1158686, -18.5077667, 18.5731659
16: -15.4804287, 6.3113661, -15.4574747, 6.2229419, -19.1705589, 19.2399673
17: -23.0655708, 1.6993694, -23.0090313, 1.7146137, -23.1635208, 23.0819016
18: 1.8119602, 23.2256718, 1.9338903, 23.2226639, -18.7704926, 18.6475525
19: -0.8278403, 11.5558300, -0.7670751, 11.5510139, -11.0246582, 10.9762268
20: -4.4422865, 9.5783911, -4.3724318, 9.5800638, -13.3152046, 13.2499962
21: -1.3676357, 15.5876322, -1.3108234, 15.5906239, -15.9739456, 15.9241180
22: -3.0698359, 11.4771376, -3.0214140, 11.4737968, -13.5289803, 13.4881020
23: -1.3615961, 15.6514025, -1.2750950, 15.6341858, -13.2294998, 13.1584778
24: -1.8925862, 16.2903309, -1.8002572, 16.2771091, -15.0339050, 14.9543533
25: -2.7067456, 16.4090519, -2.6290855, 16.4058838, -17.5075760, 17.4399719
26: -5.3951721, 21.1120186, -5.2581892, 21.1227245, -25.3659897, 25.2165451
27: -0.4226999, 15.5738392, -0.3482566, 15.5723972, -13.4388275, 13.3659286
28: -1.4754086, 15.4956417, -1.4016747, 15.4845648, -14.1216507, 14.0650520
29: -2.0507777, 12.6625614, -1.9971306, 12.6551828, -11.2327309, 11.1800995
30: -8.1343536, 14.7812119, -8.0559540, 14.7749710, -20.0615692, 19.9900665
31: 0.5321569, 16.0178909, 0.6090493, 16.0139313, -14.2697983, 14.2071381
32: -21.9960537, 1.9880285, -21.9800854, 1.9772677, -18.8840942, 18.8717728
33: -39.6672974, -10.4788313, -39.6476898, -10.5498104, -20.8332825, 20.8736954
34: -33.3761024, -10.0907688, -33.3347092, -10.1054087, -17.4376907, 17.4040642
35: -24.0535812, -0.8316960, -24.0403385, -0.8754277, -18.6880798, 18.7092209
36: -20.7974720, 5.2426414, -20.7694473, 5.2164507, -20.0344391, 20.0194702
37: -32.2936554, -2.7470789, -32.2530785, -2.7676291, -25.9036713, 25.8556137
38: -28.7824326, 0.5910354, -28.7224770, 0.5697289, -24.3888245, 24.3206024
39: -43.9327431, -10.2958393, -43.8848038, -10.3882523, -22.9373856, 22.9685020
40: -31.3292580, -13.0582123, -31.3062973, -13.0724869, -12.8834152, 12.8567390
41: -19.8928928, 2.0320132, -19.8667870, 2.0174727, -18.7426682, 18.7252579
42: -20.0959663, -3.5468354, -20.0558720, -3.5541677, -13.5440369, 13.5004387

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
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
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1583
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
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5005572, upper bound: 10.4448820
time: 22.96 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5005572, upper bound: 10.4576750
time: 23.40 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4589179
time: 23.93 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4717248
time: 19.32 seconds

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

Time for backsubstitution: 2.18 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4720691
time: 21.61 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4848774
time: 27.42 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -29.3442383, -2.3596344, -29.3263702, -2.4745684, -18.0578003, 18.1649933
1: -13.7343369, 2.6917863, -13.7254906, 2.6403651, -11.8184204, 11.8631592
2: -12.0230131, 4.0888948, -12.0179176, 4.0294471, -10.5491295, 10.6093597
3: -21.1220818, -0.8241706, -21.1220951, -0.9347596, -16.1519814, 16.2758904
4: -19.4383011, 2.7236137, -19.4239502, 2.6381540, -14.3922997, 14.4706535
5: -15.5288572, 4.2737198, -15.5220041, 4.1728821, -15.3047943, 15.4092064
6: -21.6954613, -0.4472208, -21.6640720, -0.4643860, -16.7477341, 16.6989250
7: -18.7334957, 2.4371915, -18.7228279, 2.3911071, -16.9133148, 16.9573135
8: -28.8648109, -1.3972993, -28.8565826, -1.4279346, -17.8240356, 17.8501663
9: -19.1168900, 2.5306916, -19.0981216, 2.4068863, -17.7654877, 17.8723526
10: -16.8407078, 5.4061947, -16.7893734, 5.3472915, -19.8200760, 19.8297043
11: -2.7902122, 15.8026075, -2.6412337, 15.7918549, -17.1914902, 17.0510292
12: -17.4374638, 13.1486769, -17.3223076, 13.1342068, -24.3892975, 24.2858658
13: -30.3893585, -1.6145768, -30.3530178, -1.7545309, -20.7457886, 20.8453026
14: -34.1842117, 0.3179154, -34.0209274, 0.2993498, -29.3986969, 29.2345276
15: -15.3443394, 5.2137756, -15.3309498, 5.1229138, -18.5379715, 18.6116714
16: -15.4910927, 6.2920132, -15.4575167, 6.2273822, -19.1974945, 19.2303200
17: -23.1599236, 1.7928281, -23.0156517, 1.7730260, -23.3165359, 23.1800079
18: 1.7900906, 23.2616024, 1.9279461, 23.2449646, -18.8055344, 18.6807251
19: -0.8516607, 11.5578632, -0.7730255, 11.5538063, -11.0467110, 11.0005608
20: -4.4877558, 9.5972137, -4.3811283, 9.5909081, -13.3686714, 13.2662125
21: -1.4337902, 15.6101894, -1.3196340, 15.6057405, -16.0510254, 15.9392014
22: -3.0723505, 11.4914770, -3.0274839, 11.4790554, -13.5356903, 13.5048981
23: -1.3670421, 15.6380062, -1.2792940, 15.6272860, -13.2346001, 13.1662865
24: -1.8969436, 16.2896481, -1.8052425, 16.2790661, -15.0389175, 14.9589653
25: -2.7078590, 16.4108200, -2.6363444, 16.4091187, -17.5167389, 17.4486237
26: -5.4564624, 21.1699905, -5.2672844, 21.1562672, -25.4576340, 25.2733917
27: -0.4505525, 15.5969067, -0.3574257, 15.5843859, -13.4746704, 13.3956909
28: -1.4969487, 15.5000648, -1.4072080, 15.4886742, -14.1459656, 14.0826683
29: -2.0648248, 12.6673546, -2.0014534, 12.6577311, -11.2527771, 11.1915855
30: -8.1913204, 14.8141937, -8.0652695, 14.7949820, -20.1405563, 20.0306931
31: 0.5109921, 16.0266113, 0.6008077, 16.0189304, -14.2795944, 14.2342720
32: -22.0080681, 2.0387053, -21.9940300, 1.9840426, -18.8903961, 18.9297867
33: -39.7219505, -10.4132347, -39.6829643, -10.5409737, -20.8758011, 20.9767075
34: -33.3918571, -10.0266428, -33.3569489, -10.1003599, -17.4407845, 17.4916077
35: -24.0776443, -0.8171525, -24.0563755, -0.8713155, -18.7116013, 18.7366562
36: -20.8015175, 5.2666655, -20.7747097, 5.2227688, -20.0546036, 20.0436172
37: -32.3082695, -2.7107849, -32.2664604, -2.7533054, -25.9684067, 25.8841476
38: -28.7666893, 0.6172776, -28.7297077, 0.5744519, -24.4013214, 24.3562164
39: -43.9608231, -10.2298660, -43.9071083, -10.3814316, -22.9598389, 23.0557098
40: -31.3387451, -13.0395584, -31.3175068, -13.0664635, -12.9203606, 12.8851204
41: -19.8880157, 2.0605400, -19.8769875, 2.0214159, -18.7413864, 18.7625885
42: -20.0919971, -3.5202980, -20.0646591, -3.5490806, -13.5745506, 13.5280952

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 873
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
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
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
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 972
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
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5026144, upper bound: 10.4562719
time: 25.53 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5026144, upper bound: 10.4690533
time: 23.98 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -29.3727856, -2.2887926, -29.3407249, -2.4732885, -18.0806198, 18.2514877
1: -13.7462940, 2.7308943, -13.7313271, 2.6413584, -11.8277664, 11.9087791
2: -12.0379219, 4.1356525, -12.0255642, 4.0303364, -10.5621490, 10.6640797
3: -21.1479893, -0.7508831, -21.1350365, -0.9332194, -16.1736374, 16.3640976
4: -19.4676418, 2.7832985, -19.4379272, 2.6392088, -14.4150620, 14.5428314
5: -15.5513744, 4.3395495, -15.5331345, 4.1740518, -15.3239288, 15.4873428
6: -21.7202301, -0.4364862, -21.6653385, -0.4614620, -16.7795105, 16.7095642
7: -18.7566338, 2.4835229, -18.7344933, 2.3918271, -16.9326630, 17.0157852
8: -28.8915024, -1.3464632, -28.8692188, -1.4269133, -17.8454819, 17.9138527
9: -19.1534710, 2.6035755, -19.1151943, 2.4080229, -17.7968674, 17.9623985
10: -16.8814754, 5.4361143, -16.7914314, 5.3484964, -19.8680038, 19.8583336
11: -2.8602674, 15.8355808, -2.6429739, 15.8070774, -17.2723961, 17.0773392
12: -17.5122871, 13.1782036, -17.3244820, 13.1467075, -24.4769669, 24.3169708
13: -30.4133129, -1.5532484, -30.3633499, -1.7523813, -20.7652817, 20.9182243
14: -34.2257233, 0.3389592, -34.0261917, 0.3002939, -29.4506683, 29.2600250
15: -15.3601789, 5.2595282, -15.3361197, 5.1250830, -18.5557556, 18.6623154
16: -15.5213499, 6.3504357, -15.4678345, 6.2280517, -19.2244415, 19.2988510
17: -23.2002296, 1.8151822, -23.0175190, 1.7761791, -23.3622208, 23.2030411
18: 1.7056851, 23.2876930, 1.9252625, 23.2538834, -18.8996887, 18.7046471
19: -0.8922381, 11.5628433, -0.7744999, 11.5539980, -11.0852757, 11.0122013
20: -4.5329733, 9.6020975, -4.3839836, 9.5911217, -13.4129715, 13.2789249
21: -1.4699097, 15.6179647, -1.3221583, 15.6063423, -16.0861969, 15.9548836
22: -3.1004710, 11.4949245, -3.0287454, 11.4803925, -13.5660439, 13.5162773
23: -1.4211035, 15.6555595, -1.2808442, 15.6357756, -13.2925568, 13.1784878
24: -1.9554725, 16.3124905, -1.8061724, 16.2876968, -15.1042709, 14.9777451
25: -2.7589598, 16.4193535, -2.6380348, 16.4110947, -17.5659790, 17.4598083
26: -5.5491886, 21.1841850, -5.2694902, 21.1581020, -25.5568619, 25.2915955
27: -0.4988518, 15.6117020, -0.3591566, 15.5903234, -13.5269394, 13.4082260
28: -1.5419674, 15.5093269, -1.4090238, 15.4913063, -14.1907539, 14.0963745
29: -2.0976999, 12.6855593, -2.0020235, 12.6658392, -11.2956161, 11.2057228
30: -8.2396049, 14.8338432, -8.0670700, 14.8034439, -20.1943283, 20.0490646
31: 0.4606099, 16.0319233, 0.5985441, 16.0199566, -14.3294411, 14.2491570
32: -22.0255852, 2.0418525, -21.9939156, 1.9846792, -18.9139328, 18.9342270
33: -39.7400780, -10.3667107, -39.6804924, -10.5391445, -20.9039650, 21.0172729
34: -33.4246063, -10.0184488, -33.3583832, -10.0989552, -17.4805222, 17.5039940
35: -24.0894985, -0.7910740, -24.0536079, -0.8693404, -18.7311478, 18.7610779
36: -20.8192101, 5.2708645, -20.7759209, 5.2235398, -20.0790253, 20.0482483
37: -32.3448524, -2.7111869, -32.2686920, -2.7560630, -26.0189514, 25.8930511
38: -28.8204079, 0.6295242, -28.7329597, 0.5783639, -24.4756165, 24.3701324
39: -43.9959793, -10.1878300, -43.9142838, -10.3806458, -22.9957123, 23.0989113
40: -31.3594913, -13.0358191, -31.3196373, -13.0662460, -12.9474449, 12.8901443
41: -19.9182167, 2.0728583, -19.8781605, 2.0246704, -18.7790833, 18.7768021
42: -20.1241741, -3.5113134, -20.0659447, -3.5476599, -13.6163139, 13.5380936

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 973
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
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 972
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
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1006

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5026144, upper bound: 10.4681091
time: 28.81 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5026144, upper bound: 10.4808896
time: 27.50 seconds

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

Time for backsubstitution: 2.20 seconds

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
time: 21.63 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4950896
time: 23.98 seconds

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

Time for backsubstitution: 2.20 seconds

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
time: 17.14 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5082337, upper bound: 10.5082330
time: 25.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 44.59 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5005572, upper bound: 10.4330251
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5005572, upper bound: 10.4458188
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5005572, upper bound: 10.4448820
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5005572, upper bound: 10.4576750
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4589179
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4717248
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4720691
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5061165, upper bound: 10.4848774
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5026144, upper bound: 10.4562719
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5026144, upper bound: 10.4690533
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5026144, upper bound: 10.4681091
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5026144, upper bound: 10.4808896
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4822953
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4950896
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5082337, upper bound: 10.4954366
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 44.59
Output dim: 18, lower bound: -10.5082337, upper bound: 10.5082330

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -29.2870102, -2.4586658, -29.2971058, -2.5083451, -17.9686279, 18.0299644
1: -13.7228889, 2.6541080, -13.7201490, 2.6217077, -11.7881165, 11.8190422
2: -12.0019894, 4.0382614, -12.0081978, 4.0070796, -10.5063248, 10.5451736
3: -21.0640182, -0.9247813, -21.0909882, -0.9598494, -16.0784302, 16.1421547
4: -19.3975468, 2.6466355, -19.4024544, 2.6060152, -14.3220444, 14.3672485
5: -15.4732552, 4.1828270, -15.4933815, 4.1442947, -15.2276611, 15.2882919
6: -21.6632214, -0.4729910, -21.6569901, -0.4716415, -16.6713486, 16.6585617
7: -18.6960335, 2.3954353, -18.7054043, 2.3639612, -16.8510818, 16.8960571
8: -28.8370018, -1.4268913, -28.8439140, -1.4620109, -17.7644882, 17.8060379
9: -19.0522976, 2.4212446, -19.0651741, 2.3783739, -17.6771622, 17.7325745
10: -16.7974968, 5.3399334, -16.7741451, 5.3286042, -19.7562599, 19.7427139
11: -2.6426182, 15.7410746, -2.6043921, 15.7574854, -17.0186005, 16.9639015
12: -17.3231373, 13.0456982, -17.2975197, 13.0834255, -24.2245255, 24.1625519
13: -30.3555088, -1.7062125, -30.3376541, -1.7714624, -20.6963882, 20.7445297
14: -34.0108948, 0.2262340, -33.9932251, 0.2491026, -29.1698380, 29.1258163
15: -15.2942362, 5.1278076, -15.3058977, 5.0998840, -18.4764862, 18.5135880
16: -15.4486389, 6.2507343, -15.4442062, 6.2180424, -19.1370049, 19.1652908
17: -23.0205154, 1.6758962, -22.9981613, 1.7093723, -23.1078339, 23.0433655
18: 1.9117193, 23.1992149, 1.9658766, 23.2130394, -18.6602592, 18.5938454
19: -0.7821331, 11.5507545, -0.7557783, 11.5505772, -10.9807167, 10.9547234
20: -4.3937340, 9.5723877, -4.3631821, 9.5777483, -13.2650909, 13.2294731
21: -1.3242054, 15.5794973, -1.2942743, 15.5893793, -15.9317970, 15.8967438
22: -3.0388949, 11.4732704, -3.0148056, 11.4715090, -13.4946861, 13.4709358
23: -1.2957487, 15.6335726, -1.2509756, 15.6251364, -13.1591225, 13.1235542
24: -1.8208623, 16.2673168, -1.7740660, 16.2681770, -14.9548340, 14.9097786
25: -2.6449318, 16.4003258, -2.6068063, 16.4034214, -17.4472809, 17.4083595
26: -5.2860408, 21.0974007, -5.2245469, 21.1200657, -25.2495422, 25.1663132
27: -0.3716378, 15.5585051, -0.3413234, 15.5654736, -13.3826523, 13.3475342
28: -1.4210620, 15.4858646, -1.3820186, 15.4809494, -14.0665283, 14.0331345
29: -2.0129771, 12.6440296, -1.9872189, 12.6464157, -11.1843262, 11.1569061
30: -8.0710869, 14.7612152, -8.0255527, 14.7657576, -19.9930267, 19.9449310
31: 0.5860124, 16.0121975, 0.6180587, 16.0121117, -14.2143250, 14.1852112
32: -21.9778137, 1.9775338, -21.9788933, 1.9628253, -18.8422699, 18.8566513
33: -39.6444473, -10.5262928, -39.6412811, -10.5533848, -20.7983780, 20.8231583
34: -33.3340149, -10.0998049, -33.3154144, -10.1084471, -17.3865547, 17.3726006
35: -24.0303764, -0.8588109, -24.0213852, -0.8792992, -18.6547852, 18.6614647
36: -20.7732010, 5.2380576, -20.7556515, 5.2149286, -20.0025864, 20.0017090
37: -32.2428589, -2.7470818, -32.2237816, -2.7655716, -25.8380051, 25.8186035
38: -28.7237167, 0.5777063, -28.7096558, 0.5636258, -24.3072433, 24.2955627
39: -43.8956833, -10.3384619, -43.8738747, -10.3901691, -22.8975983, 22.9197426
40: -31.3078423, -13.0622950, -31.3028336, -13.0733671, -12.8536949, 12.8479538
41: -19.8611164, 2.0179186, -19.8627319, 2.0109208, -18.7001648, 18.7064285
42: -20.0628223, -3.5591569, -20.0527992, -3.5618830, -13.4936829, 13.4845886

Time for backsubstitution: 2.20 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4317206
time: 22.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4317206
time: 20.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -29.2871513, -2.4478955, -29.3529510, -2.4767647, -17.9902382, 18.0932655
1: -13.7232456, 2.6607380, -13.7561550, 2.6408069, -11.8019562, 11.8606644
2: -12.0019760, 4.0460720, -12.0480824, 4.0290980, -10.5216179, 10.5920601
3: -21.0641136, -0.9170575, -21.1327896, -0.9334126, -16.1006775, 16.1918106
4: -19.3980331, 2.6574244, -19.4743004, 2.6355209, -14.3421059, 14.4497986
5: -15.4733582, 4.1919808, -15.5399780, 4.1723113, -15.2509995, 15.3441849
6: -21.6634445, -0.4720716, -21.6678467, -0.4578905, -16.6865540, 16.6713943
7: -18.6963158, 2.4045730, -18.7578106, 2.3889198, -16.8708801, 16.9583588
8: -28.8373604, -1.4135628, -28.9227638, -1.4281816, -17.7869492, 17.8969879
9: -19.0528278, 2.4297516, -19.1155663, 2.4054475, -17.6971703, 17.7910690
10: -16.7985992, 5.3444185, -16.8085060, 5.3498859, -19.7782784, 19.7837524
11: -2.6523967, 15.7413044, -2.6391199, 15.8190880, -17.0870399, 16.9917870
12: -17.3295403, 13.0468082, -17.3175220, 13.1355772, -24.2854004, 24.1790009
13: -30.3565025, -1.7045178, -30.3559074, -1.7460198, -20.7345123, 20.7585678
14: -34.0137138, 0.2274222, -34.0303497, 0.2650080, -29.2055511, 29.1535339
15: -15.2949133, 5.1334004, -15.3505526, 5.1227674, -18.4956970, 18.5616608
16: -15.4497643, 6.2524252, -15.4707890, 6.2310076, -19.1694107, 19.1821899
17: -23.0227661, 1.6765594, -23.0174026, 1.7589133, -23.1789398, 23.0559845
18: 1.8995461, 23.1994934, 1.9250445, 23.2713203, -18.7310638, 18.6310883
19: -0.7860689, 11.5508375, -0.7765799, 11.5655994, -11.0000000, 10.9751282
20: -4.3961973, 9.5726070, -4.3826218, 9.5849075, -13.2751617, 13.2500172
21: -1.3298526, 15.5797253, -1.3194246, 15.6113749, -15.9555435, 15.9191475
22: -3.0406628, 11.4735641, -3.0282040, 11.4800596, -13.5022850, 13.4908066
23: -1.3051176, 15.6337719, -1.2798743, 15.6699924, -13.2135239, 13.1486206
24: -1.8313913, 16.2674255, -1.8110271, 16.3222752, -15.0198936, 14.9439697
25: -2.6536136, 16.4004421, -2.6405697, 16.4493980, -17.5017853, 17.4403000
26: -5.2991700, 21.0975971, -5.2693076, 21.1713047, -25.3148193, 25.2112274
27: -0.3735957, 15.5589142, -0.3594022, 15.5811291, -13.4000626, 13.3641052
28: -1.4283948, 15.4861965, -1.4083099, 15.5209694, -14.1136169, 14.0576668
29: -2.0169392, 12.6442451, -2.0003579, 12.6650238, -11.2086945, 11.1669655
30: -8.0829172, 14.7613754, -8.0654106, 14.8399410, -20.0773163, 19.9769821
31: 0.5836649, 16.0124741, 0.6010904, 16.0186234, -14.2160110, 14.2075500
32: -21.9782524, 1.9833775, -22.0210190, 1.9904766, -18.8657684, 18.9169731
33: -39.6480103, -10.5256443, -39.6612434, -10.5339622, -20.8192825, 20.8428116
34: -33.3414803, -10.0992146, -33.3433952, -10.0554094, -17.4475174, 17.3950424
35: -24.0394840, -0.8581161, -24.0544815, -0.8209162, -18.7222824, 18.6898270
36: -20.7784538, 5.2383080, -20.7756557, 5.2521343, -20.0465164, 20.0203705
37: -32.2540588, -2.7468452, -32.2662811, -2.6990023, -25.9161530, 25.8554077
38: -28.7276726, 0.5785728, -28.7338867, 0.5966148, -24.3491745, 24.3199158
39: -43.8970947, -10.3381729, -43.9010239, -10.3829575, -22.9208984, 22.9373093
40: -31.3082829, -13.0620861, -31.3167534, -13.0692129, -12.8701439, 12.8517723
41: -19.8621178, 2.0192456, -19.8735733, 2.0327733, -18.7241669, 18.7231064
42: -20.0634766, -3.5567541, -20.0709496, -3.5459049, -13.5089226, 13.5103722

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 981
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4445204
time: 27.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4445204
time: 22.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -29.3155403, -2.3878031, -29.3114891, -2.5070157, -17.9914970, 18.1164513
1: -13.7348356, 2.6932209, -13.7259817, 2.6226933, -11.7974358, 11.8646660
2: -12.0168972, 4.0849862, -12.0158501, 4.0079532, -10.5193291, 10.5998840
3: -21.0899162, -0.8515115, -21.1039295, -0.9582782, -16.1000671, 16.2303314
4: -19.4267979, 2.7063065, -19.4164524, 2.6071172, -14.3448334, 14.4394302
5: -15.4957790, 4.2486992, -15.5044889, 4.1454573, -15.2467880, 15.3664246
6: -21.6879520, -0.4622841, -21.6582336, -0.4686975, -16.7031403, 16.6692276
7: -18.7191658, 2.4417644, -18.7170334, 2.3646884, -16.8703613, 16.9544830
8: -28.8636799, -1.3760028, -28.8565025, -1.4609985, -17.7859116, 17.8697243
9: -19.0888958, 2.4941573, -19.0822449, 2.3795280, -17.7085190, 17.8226395
10: -16.8382893, 5.3698235, -16.7761955, 5.3298244, -19.8042603, 19.7712708
11: -2.7127194, 15.7740688, -2.6061144, 15.7727365, -17.0995255, 16.9902382
12: -17.3979607, 13.0753021, -17.2997475, 13.0959759, -24.3121719, 24.1936569
13: -30.3794460, -1.6449151, -30.3479614, -1.7692676, -20.7159119, 20.8174248
14: -34.0523949, 0.2473431, -33.9985504, 0.2500248, -29.2217560, 29.1513062
15: -15.3100576, 5.1735449, -15.3110876, 5.1020193, -18.4942017, 18.5642433
16: -15.4788809, 6.3091621, -15.4545050, 6.2186770, -19.1639175, 19.2338104
17: -23.0608406, 1.6982193, -23.0000286, 1.7124939, -23.1535110, 23.0663376
18: 1.8273301, 23.2252808, 1.9632177, 23.2219486, -18.7543716, 18.6177597
19: -0.8227038, 11.5557261, -0.7572365, 11.5507879, -11.0192566, 10.9663544
20: -4.4389291, 9.5772829, -4.3660517, 9.5779486, -13.3093300, 13.2421799
21: -1.3602881, 15.5872688, -1.2967896, 15.5899973, -15.9669876, 15.9123726
22: -3.0670133, 11.4766493, -3.0160460, 11.4728765, -13.5250206, 13.4822884
23: -1.3498044, 15.6511269, -1.2525406, 15.6336346, -13.2170563, 13.1357574
24: -1.8793664, 16.2901955, -1.7750025, 16.2768059, -15.0201721, 14.9285240
25: -2.6959553, 16.4088020, -2.6084542, 16.4053802, -17.4965057, 17.4195595
26: -5.3787398, 21.1115799, -5.2267389, 21.1219292, -25.3487091, 25.1845322
27: -0.4199810, 15.5733204, -0.3430505, 15.5714073, -13.4349251, 13.3600426
28: -1.4660559, 15.4951229, -1.3838153, 15.4835796, -14.1112518, 14.0468063
29: -2.0458713, 12.6622276, -1.9877806, 12.6545200, -11.2271843, 11.1710091
30: -8.1194344, 14.7808437, -8.0274105, 14.7742424, -20.0468063, 19.9632645
31: 0.5356803, 16.0174904, 0.6158090, 16.0131493, -14.2641296, 14.2000580
32: -21.9953156, 1.9807234, -21.9787865, 1.9634318, -18.8658142, 18.8610497
33: -39.6626091, -10.4797297, -39.6387939, -10.5515995, -20.8265839, 20.8637314
34: -33.3667717, -10.0916157, -33.3168640, -10.1070833, -17.4263000, 17.3850021
35: -24.0422173, -0.8326635, -24.0186405, -0.8773892, -18.6743164, 18.6858864
36: -20.7908592, 5.2422791, -20.7568321, 5.2157125, -20.0270233, 20.0063858
37: -32.2794342, -2.7474370, -32.2260513, -2.7683125, -25.8885193, 25.8275146
38: -28.7773933, 0.5899701, -28.7128849, 0.5675478, -24.3816147, 24.3094864
39: -43.9307175, -10.2964439, -43.8810120, -10.3893805, -22.9335251, 22.9629784
40: -31.3285599, -13.0585518, -31.3049355, -13.0730972, -12.8807793, 12.8529701
41: -19.8913612, 2.0302792, -19.8638916, 2.0142233, -18.7378540, 18.7206421
42: -20.0950089, -3.5501349, -20.0540543, -3.5604706, -13.5354424, 13.4945526

Time for backsubstitution: 2.19 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 873
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4435848
time: 27.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4993307, upper bound: 10.4435848
time: 21.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -29.3156738, -2.3770866, -29.3672962, -2.4753971, -18.0130882, 18.1797600
1: -13.7352066, 2.6998634, -13.7619858, 2.6418221, -11.8112946, 11.9062767
2: -12.0168715, 4.0928125, -12.0557280, 4.0299664, -10.5346146, 10.6467533
3: -21.0900211, -0.8437834, -21.1457291, -0.9318409, -16.1223068, 16.2800140
4: -19.4273529, 2.7170854, -19.4883060, 2.6366239, -14.3648758, 14.5219727
5: -15.4958916, 4.2578764, -15.5510855, 4.1734219, -15.2701111, 15.4223442
6: -21.6881981, -0.4613647, -21.6690826, -0.4549665, -16.7183533, 16.6820068
7: -18.7194729, 2.4509020, -18.7694626, 2.3896718, -16.8901596, 17.0167465
8: -28.8640232, -1.3626771, -28.9353333, -1.4271364, -17.8083878, 17.9607201
9: -19.0894070, 2.5026762, -19.1326008, 2.4065833, -17.7285652, 17.8811493
10: -16.8393879, 5.3742957, -16.8105583, 5.3511190, -19.8262711, 19.8123474
11: -2.7224672, 15.7742939, -2.6408546, 15.8343210, -17.1679573, 17.0181389
12: -17.4043198, 13.0763693, -17.3196964, 13.1480808, -24.3731003, 24.2101669
13: -30.3804188, -1.6431561, -30.3662376, -1.7437658, -20.7540131, 20.8314972
14: -34.0552521, 0.2485065, -34.0357246, 0.2659473, -29.2575073, 29.1790161
15: -15.3107367, 5.1791477, -15.3557148, 5.1248927, -18.5134125, 18.6123085
16: -15.4799957, 6.3108377, -15.4810944, 6.2316647, -19.1963501, 19.2507401
17: -23.0630989, 1.6989295, -23.0192947, 1.7620883, -23.2245712, 23.0789719
18: 1.8151197, 23.2255516, 1.9223723, 23.2802086, -18.8252029, 18.6549873
19: -0.8266401, 11.5557919, -0.7780561, 11.5658035, -11.0385380, 10.9867802
20: -4.4413924, 9.5774660, -4.3854895, 9.5851173, -13.3194122, 13.2627716
21: -1.3659430, 15.5874882, -1.3219466, 15.6119518, -15.9906960, 15.9348068
22: -3.0687904, 11.4769611, -3.0294430, 11.4813967, -13.5326080, 13.5021667
23: -1.3591280, 15.6513414, -1.2814240, 15.6784611, -13.2714691, 13.1608067
24: -1.8899212, 16.2902794, -1.8119712, 16.3308907, -15.0852318, 14.9627304
25: -2.7046351, 16.4089336, -2.6422353, 16.4513817, -17.5510330, 17.4515305
26: -5.3918905, 21.1118164, -5.2715316, 21.1731644, -25.4139862, 25.2294617
27: -0.4219055, 15.5737104, -0.3611197, 15.5870752, -13.4523544, 13.3766403
28: -1.4734106, 15.4954872, -1.4101276, 15.5235996, -14.1583481, 14.0713501
29: -2.0498405, 12.6624584, -2.0009141, 12.6731339, -11.2515488, 11.1810684
30: -8.1312408, 14.7809868, -8.0672426, 14.8484278, -20.1310730, 19.9953156
31: 0.5332880, 16.0177593, 0.5988069, 16.0196819, -14.2658310, 14.2224159
32: -21.9957695, 1.9865537, -22.0208855, 1.9911318, -18.8893204, 18.9214020
33: -39.6661758, -10.4791260, -39.6587982, -10.5320988, -20.8474655, 20.8833504
34: -33.3742332, -10.0910110, -33.3448639, -10.0540199, -17.4872665, 17.4074478
35: -24.0513401, -0.8319776, -24.0517502, -0.8189840, -18.7418365, 18.7142372
36: -20.7961597, 5.2424841, -20.7768764, 5.2529497, -20.0709381, 20.0250626
37: -32.2907104, -2.7471929, -32.2685394, -2.7017808, -25.9666290, 25.8643341
38: -28.7813969, 0.5907669, -28.7370930, 0.6005116, -24.4235458, 24.3338699
39: -43.9322624, -10.2960758, -43.9081383, -10.3821707, -22.9568405, 22.9804802
40: -31.3290195, -13.0583124, -31.3188667, -13.0689430, -12.8971939, 12.8567810
41: -19.8923569, 2.0315993, -19.8747311, 2.0360363, -18.7618484, 18.7372742
42: -20.0956612, -3.5477538, -20.0721893, -3.5444822, -13.5506744, 13.5203514

Time for backsubstitution: 2.20 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 873
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4563868
time: 30.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4563868
time: 24.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -29.2916679, -2.4038367, -29.3283215, -2.4092593, -18.0518265, 18.1179886
1: -13.7266121, 2.6874273, -13.7363796, 2.6826992, -11.8424034, 11.8676338
2: -12.0041704, 4.0695744, -12.0195570, 4.0643978, -10.5591812, 10.5892239
3: -21.0674210, -0.8844686, -21.1042442, -0.8855710, -16.1509171, 16.1975098
4: -19.4022751, 2.6884289, -19.4307365, 2.6816068, -14.3876648, 14.4418716
5: -15.4766026, 4.2250938, -15.5101089, 4.2212429, -15.2993546, 15.3470688
6: -21.6746864, -0.4624720, -21.6803513, -0.4511013, -16.7075729, 16.6937218
7: -18.6992702, 2.4272928, -18.7238388, 2.4223223, -16.9010315, 16.9437408
8: -28.8444595, -1.4012489, -28.8659248, -1.4140339, -17.8091431, 17.8489990
9: -19.0582523, 2.4748631, -19.0986710, 2.4736037, -17.7617264, 17.8245010
10: -16.8427467, 5.3554630, -16.8546810, 5.3593607, -19.8344193, 19.8387032
11: -2.6905260, 15.7465553, -2.6914229, 15.7843666, -17.0997734, 17.0457764
12: -17.3554821, 13.0564623, -17.3577156, 13.1065311, -24.2832108, 24.2314301
13: -30.3639278, -1.6110888, -30.3895588, -1.6015568, -20.8499374, 20.8901787
14: -34.0523949, 0.2405462, -34.0719299, 0.2749815, -29.2405701, 29.2213821
15: -15.3055382, 5.1649656, -15.3314667, 5.1677551, -18.5559387, 18.5787010
16: -15.4595509, 6.2959661, -15.4767895, 6.2991605, -19.2174759, 19.2419968
17: -23.0544434, 1.6931493, -23.0604248, 1.7459371, -23.1810455, 23.1199646
18: 1.8472447, 23.2058067, 1.8495641, 23.2390118, -18.7566299, 18.7121162
19: -0.8162355, 11.5532055, -0.8190880, 11.5564032, -11.0274639, 11.0263977
20: -4.4399290, 9.5752182, -4.4489059, 9.5841951, -13.3184395, 13.3185501
21: -1.3643923, 15.5814419, -1.3691273, 15.5939264, -15.9791183, 15.9725838
22: -3.0747855, 11.4764996, -3.0803521, 11.4832230, -13.5446815, 13.5375252
23: -1.3449831, 15.6358833, -1.3403244, 15.6437359, -13.2308540, 13.2057171
24: -1.8708105, 16.2714806, -1.8658733, 16.2897034, -15.0307007, 14.9959030
25: -2.6891546, 16.4036980, -2.6894503, 16.4114952, -17.5052567, 17.4926071
26: -5.3575454, 21.1042290, -5.3556485, 21.1395187, -25.3470688, 25.3034668
27: -0.4211354, 15.5628452, -0.4306641, 15.5874014, -13.4547195, 13.4368019
28: -1.4664502, 15.4887943, -1.4653006, 15.4973726, -14.1311913, 14.1091423
29: -2.0451174, 12.6468735, -2.0450299, 12.6644630, -11.2408943, 11.2064819
30: -8.1228428, 14.7649555, -8.1201668, 14.7933550, -20.0765381, 20.0266571
31: 0.5397387, 16.0145359, 0.5323977, 16.0201797, -14.2728500, 14.2726669
32: -21.9888268, 1.9959807, -22.0017300, 1.9975996, -18.8893356, 18.9033012
33: -39.6650162, -10.4778175, -39.6919861, -10.4670820, -20.8955383, 20.9281120
34: -33.3722229, -10.0926514, -33.3842316, -10.0851946, -17.4513855, 17.4401665
35: -24.0442619, -0.8256247, -24.0508938, -0.8181438, -18.7298126, 18.7289963
36: -20.7845345, 5.2751751, -20.7877197, 5.2816668, -20.0678711, 20.0686188
37: -32.2622948, -2.7067375, -32.2668610, -2.6944799, -25.9216156, 25.9051819
38: -28.7436619, 0.6102405, -28.7524509, 0.6220288, -24.3761749, 24.3682327
39: -43.9183655, -10.2635021, -43.9492607, -10.2579660, -23.0260773, 23.0791168
40: -31.3205185, -13.0404406, -31.3343716, -13.0351801, -12.8944855, 12.9064369
41: -19.8690319, 2.0340490, -19.8788338, 2.0418608, -18.7434235, 18.7453384
42: -20.0834751, -3.5519776, -20.0915031, -3.5462422, -13.5471268, 13.5400238

Time for backsubstitution: 2.20 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4869595, upper bound: 10.4575747
time: 9.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5048110, upper bound: 10.4575747
time: 23.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -29.2918167, -2.3931351, -29.3841267, -2.3776803, -18.0734406, 18.1813049
1: -13.7269859, 2.6940603, -13.7723827, 2.7018259, -11.8562660, 11.9092484
2: -12.0041580, 4.0773840, -12.0594339, 4.0864315, -10.5744705, 10.6360970
3: -21.0675144, -0.8767605, -21.1460495, -0.8591256, -16.1731567, 16.2471695
4: -19.4028072, 2.6992502, -19.5026264, 2.7110448, -14.4077034, 14.5244370
5: -15.4767313, 4.2342768, -15.5567160, 4.2492452, -15.3227196, 15.4029579
6: -21.6748981, -0.4615240, -21.6911678, -0.4373455, -16.7227859, 16.7065468
7: -18.6995678, 2.4364533, -18.7762299, 2.4473128, -16.9208832, 17.0060196
8: -28.8447800, -1.3879371, -28.9447632, -1.3801837, -17.8316116, 17.9399872
9: -19.0587654, 2.4834030, -19.1490593, 2.5006380, -17.7817535, 17.8829651
10: -16.8438187, 5.3599210, -16.8890533, 5.3806190, -19.8564262, 19.8797188
11: -2.7003043, 15.7467909, -2.7261548, 15.8459673, -17.1682053, 17.0736313
12: -17.3618736, 13.0575380, -17.3776855, 13.1586847, -24.3441086, 24.2478790
13: -30.3650208, -1.6092906, -30.4078064, -1.5760436, -20.8880615, 20.9042892
14: -34.0552292, 0.2417545, -34.1090431, 0.2909079, -29.2763290, 29.2490463
15: -15.3062210, 5.1705990, -15.3761377, 5.1906090, -18.5751266, 18.6267624
16: -15.4606771, 6.2976313, -15.5033731, 6.3121538, -19.2499046, 19.2589264
17: -23.0567131, 1.6938612, -23.0796623, 1.7955451, -23.2521133, 23.1326218
18: 1.8350611, 23.2060699, 1.8087559, 23.2973022, -18.8274307, 18.7493515
19: -0.8201742, 11.5532637, -0.8399315, 11.5714207, -11.0467548, 11.0468121
20: -4.4423785, 9.5754156, -4.4683228, 9.5913334, -13.3285179, 13.3391190
21: -1.3700581, 15.5817013, -1.3942947, 15.6158905, -16.0028458, 15.9949722
22: -3.0765803, 11.4767971, -3.0937161, 11.4917679, -13.5522537, 13.5574112
23: -1.3543081, 15.6360912, -1.3692188, 15.6885815, -13.2852821, 13.2307739
24: -1.8813453, 16.2715721, -1.9028459, 16.3437729, -15.0957642, 15.0301476
25: -2.6978183, 16.4038048, -2.7232251, 16.4575005, -17.5597992, 17.5245857
26: -5.3707027, 21.1044483, -5.4004316, 21.1907654, -25.4123230, 25.3483734
27: -0.4230766, 15.5632544, -0.4486766, 15.6030426, -13.4720993, 13.4534035
28: -1.4737983, 15.4891510, -1.4915805, 15.5373964, -14.1782951, 14.1336823
29: -2.0490797, 12.6470957, -2.0581689, 12.6830559, -11.2652321, 11.2165489
30: -8.1346493, 14.7650785, -8.1599684, 14.8675213, -20.1607819, 20.0586548
31: 0.5373626, 16.0148125, 0.5153947, 16.0267105, -14.2745628, 14.2950096
32: -21.9892883, 2.0017824, -22.0438881, 2.0252967, -18.9128418, 18.9636078
33: -39.6685677, -10.4772024, -39.7119217, -10.4476357, -20.9163895, 20.9476776
34: -33.3796654, -10.0920191, -33.4122391, -10.0321522, -17.5123482, 17.4625626
35: -24.0533791, -0.8249478, -24.0839157, -0.7597656, -18.7972946, 18.7572708
36: -20.7898006, 5.2754140, -20.8076878, 5.3189192, -20.1117859, 20.0872421
37: -32.2735519, -2.7064571, -32.3093033, -2.6279340, -25.9997940, 25.9419403
38: -28.7476273, 0.6111236, -28.7765598, 0.6549735, -24.4181442, 24.3926392
39: -43.9198418, -10.2631502, -43.9764557, -10.2507048, -23.0493927, 23.0966644
40: -31.3209839, -13.0401926, -31.3482857, -13.0310211, -12.9109383, 12.9102287
41: -19.8700600, 2.0353947, -19.8896236, 2.0636775, -18.7674637, 18.7620544
42: -20.0841198, -3.5495744, -20.1096306, -3.5302608, -13.5623360, 13.5658150

Time for backsubstitution: 2.21 seconds

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
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1760
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4869595, upper bound: 10.4703865
time: 24.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5048110, upper bound: 10.4703865
time: 21.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -29.3200951, -2.3329201, -29.3426266, -2.4079804, -18.0746078, 18.2044983
1: -13.7384949, 2.7265599, -13.7421474, 2.6836805, -11.8517456, 11.9132156
2: -12.0190582, 4.1163325, -12.0272026, 4.0652447, -10.5721397, 10.6439209
3: -21.0932636, -0.8112411, -21.1171875, -0.8841195, -16.1724319, 16.2856674
4: -19.4315357, 2.7480721, -19.4447460, 2.6826367, -14.4104042, 14.5139656
5: -15.4991140, 4.2909365, -15.5212345, 4.2223592, -15.3183670, 15.4251976
6: -21.6994362, -0.4520445, -21.6815586, -0.4483223, -16.7393265, 16.7042618
7: -18.7223587, 2.4736242, -18.7354355, 2.4229889, -16.9208603, 17.0019836
8: -28.8710556, -1.3504090, -28.8785000, -1.4130201, -17.8295364, 17.9128113
9: -19.0947552, 2.5477858, -19.1156883, 2.4747248, -17.7929688, 17.9144821
10: -16.8834305, 5.3854704, -16.8567085, 5.3606143, -19.8823051, 19.8682098
11: -2.7605412, 15.7794027, -2.6930356, 15.7995882, -17.1805801, 17.0718231
12: -17.4303589, 13.0859804, -17.3598442, 13.1190166, -24.3708725, 24.2624817
13: -30.3875351, -1.5497570, -30.3995857, -1.5993686, -20.8691559, 20.9629364
14: -34.0932045, 0.2616248, -34.0771332, 0.2758765, -29.2919159, 29.2470245
15: -15.3211937, 5.2107382, -15.3365173, 5.1698389, -18.5734634, 18.6292572
16: -15.4896936, 6.3543754, -15.4870253, 6.2997704, -19.2442131, 19.3104630
17: -23.0947495, 1.7154295, -23.0622177, 1.7485662, -23.2260818, 23.1431274
18: 1.7628131, 23.2316799, 1.8469615, 23.2478943, -18.8505249, 18.7359695
19: -0.8567934, 11.5580969, -0.8204966, 11.5566597, -11.0659485, 11.0380402
20: -4.4851713, 9.5800447, -4.4517145, 9.5843525, -13.3626938, 13.3312187
21: -1.4005132, 15.5892324, -1.3716264, 15.5944958, -16.0143356, 15.9883881
22: -3.1029482, 11.4798517, -3.0815713, 11.4845486, -13.5749588, 13.5490417
23: -1.3989668, 15.6532431, -1.3418312, 15.6521492, -13.2883949, 13.2181206
24: -1.9293008, 16.2941856, -1.8666930, 16.2983475, -15.0955963, 15.0143585
25: -2.7401295, 16.4120293, -2.6909976, 16.4134712, -17.5542297, 17.5034904
26: -5.4502201, 21.1182957, -5.3577404, 21.1413879, -25.4459000, 25.3216400
27: -0.4694886, 15.5775318, -0.4323602, 15.5932655, -13.5069084, 13.4491463
28: -1.5114775, 15.4979248, -1.4670768, 15.5000286, -14.1756744, 14.1229858
29: -2.0780063, 12.6650352, -2.0455127, 12.6725578, -11.2833252, 11.2208977
30: -8.1711369, 14.7844906, -8.1218224, 14.8017960, -20.1297760, 20.0447083
31: 0.4893537, 16.0197525, 0.5301399, 16.0212421, -14.3231010, 14.2875061
32: -22.0063438, 1.9994879, -22.0017357, 1.9980631, -18.9128723, 18.9083023
33: -39.6828041, -10.4317713, -39.6884270, -10.4652653, -20.9227295, 20.9687347
34: -33.4050598, -10.0845318, -33.3856812, -10.0837975, -17.4911194, 17.4524193
35: -24.0559711, -0.7996318, -24.0488892, -0.8162093, -18.7493820, 18.7537994
36: -20.8021660, 5.2793951, -20.7889328, 5.2820826, -20.0921326, 20.0732117
37: -32.2990723, -2.7086239, -32.2692108, -2.6984286, -25.9717407, 25.9119186
38: -28.7973557, 0.6211329, -28.7556801, 0.6242533, -24.4505310, 24.3823547
39: -43.9532585, -10.2215433, -43.9565010, -10.2571888, -23.0619507, 23.1213913
40: -31.3410892, -13.0367165, -31.3365898, -13.0349455, -12.9231491, 12.9123688
41: -19.8992538, 2.0462477, -19.8799248, 2.0450132, -18.7812271, 18.7596130
42: -20.1156769, -3.5430388, -20.0927010, -3.5448997, -13.5883064, 13.5501480

Time for backsubstitution: 2.19 seconds

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
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 870
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
type: A, layer: 1, pos: 1698
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4869595, upper bound: 10.4707150
time: 24.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5048110, upper bound: 10.4707150
time: 19.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -29.3202763, -2.3221931, -29.3984623, -2.3763733, -18.0962067, 18.2678070
1: -13.7388744, 2.7331736, -13.7781448, 2.7027829, -11.8656197, 11.9548302
2: -12.0190392, 4.1241283, -12.0670805, 4.0872583, -10.5874481, 10.6908035
3: -21.0933609, -0.8035064, -21.1590042, -0.8576694, -16.1946869, 16.3353615
4: -19.4320297, 2.7589083, -19.5165939, 2.7121286, -14.4304428, 14.5965271
5: -15.4992027, 4.3001242, -15.5678263, 4.2503266, -15.3417473, 15.4811096
6: -21.6996784, -0.4511223, -21.6924267, -0.4345732, -16.7544937, 16.7170830
7: -18.7226524, 2.4827456, -18.7878723, 2.4479656, -16.9406586, 17.0642471
8: -28.8714046, -1.3371172, -28.9573212, -1.3792095, -17.8520203, 18.0037689
9: -19.0952721, 2.5563362, -19.1660805, 2.5017850, -17.8129730, 17.9729958
10: -16.8845158, 5.3899341, -16.8910828, 5.3818603, -19.9043274, 19.9092522
11: -2.7703378, 15.7796259, -2.7277923, 15.8611879, -17.2490082, 17.0996933
12: -17.4367256, 13.0870171, -17.3798218, 13.1711683, -24.4317856, 24.2789459
13: -30.3885746, -1.5479822, -30.4178619, -1.5739317, -20.9073181, 20.9770393
14: -34.0960503, 0.2628288, -34.1142998, 0.2917924, -29.3276825, 29.2746353
15: -15.3218613, 5.2163534, -15.3811922, 5.1927242, -18.5926666, 18.6773262
16: -15.4908314, 6.3560500, -15.5136166, 6.3127780, -19.2766571, 19.3273888
17: -23.0970154, 1.7161062, -23.0814629, 1.7981331, -23.2972488, 23.1557617
18: 1.7506542, 23.2319698, 1.8061223, 23.3061790, -18.9213181, 18.7731819
19: -0.8607359, 11.5581751, -0.8413439, 11.5716820, -11.0852280, 11.0584831
20: -4.4876156, 9.5802498, -4.4711246, 9.5915108, -13.3727570, 13.3517914
21: -1.4061909, 15.5894451, -1.3967810, 15.6164951, -16.0380707, 16.0108337
22: -3.1047311, 11.4801493, -3.0949404, 11.4930811, -13.5825577, 13.5689125
23: -1.4083328, 15.6534586, -1.3707409, 15.6969891, -13.3428192, 13.2431812
24: -1.9398198, 16.2942867, -1.9036980, 16.3524551, -15.1606674, 15.0486069
25: -2.7487984, 16.4121552, -2.7248116, 16.4594650, -17.6087570, 17.5354919
26: -5.4633594, 21.1185341, -5.4025378, 21.1926174, -25.5111771, 25.3665695
27: -0.4714694, 15.5779381, -0.4504080, 15.6089134, -13.5243073, 13.4657669
28: -1.5188341, 15.4983177, -1.4933715, 15.5400238, -14.2227974, 14.1475639
29: -2.0819221, 12.6652460, -2.0586541, 12.6911583, -11.3076820, 11.2309570
30: -8.1829433, 14.7846889, -8.1616707, 14.8759727, -20.2140884, 20.0767288
31: 0.4869652, 16.0200100, 0.5131450, 16.0277481, -14.3248024, 14.3098755
32: -22.0067825, 2.0053306, -22.0438442, 2.0257716, -18.9363937, 18.9686012
33: -39.6863632, -10.4311066, -39.7083664, -10.4458427, -20.9435768, 20.9883347
34: -33.4125290, -10.0839291, -33.4136124, -10.0307426, -17.5520706, 17.4748230
35: -24.0650749, -0.7989731, -24.0819530, -0.7578311, -18.8168640, 18.7820969
36: -20.8074837, 5.2796731, -20.8089180, 5.3192749, -20.1360626, 20.0917892
37: -32.3102798, -2.7083035, -32.3116226, -2.6318884, -26.0498581, 25.9486771
38: -28.8013344, 0.6219726, -28.7798004, 0.6572170, -24.4924240, 24.4067383
39: -43.9547958, -10.2211590, -43.9836807, -10.2499599, -23.0852509, 23.1389465
40: -31.3415680, -13.0365019, -31.3505344, -13.0307760, -12.9395866, 12.9161644
41: -19.9002686, 2.0475860, -19.8907661, 2.0668392, -18.8052597, 18.7762756
42: -20.1163292, -3.5406744, -20.1108208, -3.5289173, -13.6035576, 13.5759392

Time for backsubstitution: 2.20 seconds

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
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 870
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
type: A, layer: 1, pos: 1698
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
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4869595, upper bound: 10.4835275
time: 37.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5048110, upper bound: 10.4835275
time: 19.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -29.3438587, -2.3731422, -29.3255234, -2.5003920, -18.0339622, 18.1519890
1: -13.7338352, 2.6834767, -13.7245531, 2.6244874, -11.8034515, 11.8545036
2: -12.0229511, 4.0791650, -12.0177717, 4.0108695, -10.5313416, 10.5997200
3: -21.1219425, -0.8338823, -21.1217918, -0.9532800, -16.1331673, 16.2657280
4: -19.4376526, 2.7101345, -19.4226646, 2.6123710, -14.3679504, 14.4566345
5: -15.5286293, 4.2622738, -15.5215712, 4.1510854, -15.2826157, 15.3972969
6: -21.6950493, -0.4484940, -21.6632595, -0.4667740, -16.7443314, 16.6961288
7: -18.7329445, 2.4259224, -18.7218018, 2.3695378, -16.8909683, 16.9448776
8: -28.8642311, -1.4139409, -28.8554745, -1.4597158, -17.7937393, 17.8331451
9: -19.1162167, 2.5199525, -19.0968475, 2.3863432, -17.7449913, 17.8603821
10: -16.8392105, 5.4004092, -16.7864838, 5.3362970, -19.8072166, 19.8207893
11: -2.7777197, 15.8021708, -2.6173460, 15.7910061, -17.1795731, 17.0295944
12: -17.4296455, 13.1472778, -17.3073082, 13.1314459, -24.3787079, 24.2705002
13: -30.3880234, -1.6170654, -30.3504601, -1.7593198, -20.7394257, 20.8397980
14: -34.1803551, 0.3159151, -34.0135880, 0.2955570, -29.3878021, 29.2220230
15: -15.3433170, 5.2064643, -15.3290234, 5.1090841, -18.5243988, 18.6027336
16: -15.4895620, 6.2898002, -15.4545736, 6.2231503, -19.1908340, 19.2241554
17: -23.1551743, 1.7917209, -23.0066433, 1.7709577, -23.3065186, 23.1644745
18: 1.8054485, 23.2612190, 1.9572310, 23.2442360, -18.7894669, 18.6509705
19: -0.8465161, 11.5577602, -0.7631841, 11.5535679, -11.0412960, 10.9906940
20: -4.4844007, 9.5961094, -4.3747005, 9.5887814, -13.3627968, 13.2583923
21: -1.4264431, 15.6098499, -1.3055997, 15.6050930, -16.0440521, 15.9274750
22: -3.0695610, 11.4909925, -3.0221360, 11.4781408, -13.5317307, 13.4990654
23: -1.3552027, 15.6376982, -1.2567530, 15.6267185, -13.2221336, 13.1435585
24: -1.8837237, 16.2894669, -1.7799840, 16.2788010, -15.0252151, 14.9331627
25: -2.6970787, 16.4105701, -2.6157470, 16.4086494, -17.5056381, 17.4281769
26: -5.4399981, 21.1695595, -5.2358270, 21.1555004, -25.4403381, 25.2413712
27: -0.4478292, 15.5963755, -0.3522186, 15.5834141, -13.4707870, 13.3898315
28: -1.4876094, 15.4995680, -1.3893814, 15.4876642, -14.1355667, 14.0644569
29: -2.0599141, 12.6669846, -1.9920964, 12.6570663, -11.2472229, 11.1824989
30: -8.1763611, 14.8138065, -8.0366993, 14.7942238, -20.1257477, 20.0038834
31: 0.5145364, 16.0262165, 0.6075315, 16.0181503, -14.2739449, 14.2272148
32: -22.0073872, 2.0313821, -21.9926872, 1.9702001, -18.8721161, 18.9190903
33: -39.7172623, -10.4141731, -39.6740608, -10.5427618, -20.8690720, 20.9667053
34: -33.3825340, -10.0275011, -33.3391228, -10.1020222, -17.4293823, 17.4725914
35: -24.0662975, -0.8181860, -24.0346603, -0.8731961, -18.6978226, 18.7133141
36: -20.7949867, 5.2662735, -20.7621403, 5.2219887, -20.0472336, 20.0305328
37: -32.2941208, -2.7111487, -32.2394257, -2.7539768, -25.9532242, 25.8560715
38: -28.7617035, 0.6161747, -28.7201653, 0.5723381, -24.3941269, 24.3450470
39: -43.9588585, -10.2305079, -43.9033890, -10.3825293, -22.9559784, 23.0501823
40: -31.3380699, -13.0398636, -31.3161621, -13.0671253, -12.9177284, 12.8813286
41: -19.8864517, 2.0587864, -19.8740711, 2.0181468, -18.7365646, 18.7579346
42: -20.0910435, -3.5236025, -20.0628281, -3.5553689, -13.5659714, 13.5222168

Time for backsubstitution: 2.28 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4835822, upper bound: 10.4550365
time: 25.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5014215, upper bound: 10.4550365
time: 28.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -29.3440552, -2.3624258, -29.3814354, -2.4688330, -18.0555458, 18.2153091
1: -13.7342014, 2.6900940, -13.7605534, 2.6436038, -11.8173027, 11.8961105
2: -12.0229492, 4.0870090, -12.0576401, 4.0328813, -10.5466309, 10.6465874
3: -21.1220207, -0.8260918, -21.1635914, -0.9268713, -16.1553993, 16.3154030
4: -19.4381618, 2.7209334, -19.4945202, 2.6418447, -14.3879852, 14.5392036
5: -15.5287285, 4.2714329, -15.5681534, 4.1790361, -15.3059769, 15.4531822
6: -21.6952591, -0.4475479, -21.6740971, -0.4530249, -16.7595215, 16.7089272
7: -18.7332535, 2.4350553, -18.7742615, 2.3945141, -16.9107513, 17.0071869
8: -28.8645725, -1.4006453, -28.9343472, -1.4258771, -17.8162079, 17.9241333
9: -19.1167355, 2.5284560, -19.1472130, 2.4133697, -17.7650070, 17.9188766
10: -16.8402939, 5.4048786, -16.8208656, 5.3575411, -19.8292236, 19.8618507
11: -2.7875137, 15.8023815, -2.6520877, 15.8525963, -17.2480164, 17.0574799
12: -17.4360371, 13.1483307, -17.3272724, 13.1836042, -24.4395752, 24.2869568
13: -30.3890858, -1.6153316, -30.3686924, -1.7337661, -20.7775726, 20.8538628
14: -34.1831436, 0.3170748, -34.0507126, 0.3114958, -29.4236145, 29.2497635
15: -15.3439865, 5.2120743, -15.3736458, 5.1319709, -18.5435867, 18.6508179
16: -15.4906559, 6.2914782, -15.4811659, 6.2361293, -19.2232895, 19.2410965
17: -23.1574707, 1.7923856, -23.0258770, 1.8205678, -23.3776321, 23.1770935
18: 1.7932639, 23.2614994, 1.9164457, 23.3025532, -18.8602829, 18.6881561
19: -0.8504410, 11.5578327, -0.7839899, 11.5685825, -11.0605888, 11.0111046
20: -4.4868755, 9.5963097, -4.3941522, 9.5959702, -13.3728981, 13.2789803
21: -1.4321218, 15.6100540, -1.3307691, 15.6271086, -16.0678177, 15.9498787
22: -3.0713353, 11.4912910, -3.0355279, 11.4866972, -13.5393066, 13.5189590
23: -1.3645706, 15.6379223, -1.2856236, 15.6715927, -13.2765503, 13.1686115
24: -1.8942776, 16.2895660, -1.8169165, 16.3328533, -15.0902863, 14.9673538
25: -2.7057652, 16.4106941, -2.6494799, 16.4546394, -17.5601501, 17.4601326
26: -5.4531798, 21.1698151, -5.2805386, 21.2066994, -25.5056152, 25.2862625
27: -0.4497800, 15.5967617, -0.3702579, 15.5990887, -13.4881973, 13.4064293
28: -1.4949393, 15.4998941, -1.4156423, 15.5276756, -14.1826820, 14.0889969
29: -2.0638685, 12.6672173, -2.0052390, 12.6756821, -11.2716141, 11.1925545
30: -8.1882010, 14.8139553, -8.0765343, 14.8684549, -20.2100830, 20.0359192
31: 0.5121489, 16.0264664, 0.5905652, 16.0246391, -14.2756729, 14.2495575
32: -22.0078697, 2.0372047, -22.0348492, 1.9978490, -18.8955917, 18.9794540
33: -39.7207718, -10.4135475, -39.6940651, -10.5232716, -20.8899994, 20.9863510
34: -33.3899765, -10.0269203, -33.3670959, -10.0489779, -17.4903336, 17.4950066
35: -24.0754013, -0.8174815, -24.0677299, -0.8148041, -18.7653198, 18.7417488
36: -20.8002205, 5.2665110, -20.7821198, 5.2591791, -20.0911484, 20.0491638
37: -32.3052826, -2.7108698, -32.2819138, -2.6874008, -26.0313416, 25.8928909
38: -28.7656765, 0.6170449, -28.7443256, 0.6053147, -24.4360199, 24.3694000
39: -43.9603348, -10.2301130, -43.9305038, -10.3753119, -22.9792404, 23.0677643
40: -31.3384972, -13.0396481, -31.3301277, -13.0629597, -12.9341469, 12.8851700
41: -19.8874760, 2.0601084, -19.8848801, 2.0400126, -18.7605438, 18.7746811
42: -20.0917053, -3.5212150, -20.0809917, -3.5393829, -13.5812645, 13.5480194

Time for backsubstitution: 2.22 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 895
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
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4835822, upper bound: 10.4678220
time: 29.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5014215, upper bound: 10.4678219
time: 24.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -29.3723831, -2.3023610, -29.3399353, -2.4991093, -18.0568237, 18.2384872
1: -13.7458019, 2.7225866, -13.7304020, 2.6254826, -11.8127937, 11.9001312
2: -12.0378456, 4.1259131, -12.0254326, 4.0117369, -10.5443573, 10.6544418
3: -21.1478329, -0.7605920, -21.1347389, -0.9517622, -16.1548157, 16.3539238
4: -19.4669647, 2.7697968, -19.4366531, 2.6134329, -14.3907318, 14.5288353
5: -15.5511484, 4.3281360, -15.5327177, 4.1522408, -15.3017273, 15.4754372
6: -21.7197533, -0.4377775, -21.6645279, -0.4638157, -16.7761002, 16.7067566
7: -18.7560921, 2.4722471, -18.7334766, 2.3702736, -16.9103088, 17.0033264
8: -28.8908768, -1.3630648, -28.8681278, -1.4587097, -17.8152008, 17.8968277
9: -19.1527977, 2.5928307, -19.1139145, 2.3874869, -17.7763748, 17.9504166
10: -16.8800068, 5.4302945, -16.7885647, 5.3375225, -19.8551903, 19.8493881
11: -2.8477905, 15.8351431, -2.6191008, 15.8062496, -17.2604675, 17.0559158
12: -17.5044651, 13.1767368, -17.3095322, 13.1439323, -24.4663544, 24.3016052
13: -30.4119759, -1.5557227, -30.3607883, -1.7571082, -20.7589111, 20.9126740
14: -34.2219009, 0.3369484, -34.0188446, 0.2964544, -29.4398041, 29.2475433
15: -15.3591585, 5.2521906, -15.3341551, 5.1112242, -18.5421448, 18.6533546
16: -15.5198097, 6.3482113, -15.4648781, 6.2237720, -19.2178040, 19.2927132
17: -23.1955204, 1.8141057, -23.0085106, 1.7740922, -23.3521729, 23.1874771
18: 1.7210307, 23.2873154, 1.9545441, 23.2531490, -18.8836212, 18.6748695
19: -0.8870878, 11.5627136, -0.7646437, 11.5537758, -11.0798492, 11.0023270
20: -4.5296278, 9.6009932, -4.3775644, 9.5889988, -13.4070702, 13.2711353
21: -1.4625621, 15.6176205, -1.3081350, 15.6056948, -16.0792198, 15.9431419
22: -3.0976751, 11.4944382, -3.0233850, 11.4794827, -13.5620499, 13.5104370
23: -1.4092836, 15.6552668, -1.2583036, 15.6352320, -13.2801056, 13.1557674
24: -1.9422808, 16.3123474, -1.7809463, 16.2873859, -15.0905724, 14.9519196
25: -2.7481537, 16.4191093, -2.6174283, 16.4106369, -17.5548859, 17.4394073
26: -5.5327082, 21.1837692, -5.2380419, 21.1573524, -25.5395584, 25.2596054
27: -0.4961100, 15.6111965, -0.3539228, 15.5893478, -13.5230637, 13.4023552
28: -1.5326471, 15.5088005, -1.3911762, 15.4903040, -14.1803322, 14.0781441
29: -2.0928247, 12.6852083, -1.9926214, 12.6651649, -11.2900696, 11.1966248
30: -8.2246952, 14.8334274, -8.0385580, 14.8027563, -20.1795120, 20.0222244
31: 0.4641466, 16.0315113, 0.6052699, 16.0191898, -14.3237839, 14.2420845
32: -22.0248528, 2.0345550, -21.9925308, 1.9708452, -18.8956909, 18.9234695
33: -39.7353668, -10.3676138, -39.6716423, -10.5409489, -20.8972092, 21.0072937
34: -33.4152718, -10.0192976, -33.3405914, -10.1006346, -17.4691162, 17.4850006
35: -24.0781364, -0.7920399, -24.0319424, -0.8712621, -18.7173767, 18.7377625
36: -20.8126411, 5.2705059, -20.7633076, 5.2227612, -20.0716553, 20.0351639
37: -32.3307343, -2.7115092, -32.2416725, -2.7567577, -26.0038147, 25.8649902
38: -28.8154049, 0.6283841, -28.7233620, 0.5762444, -24.4684296, 24.3590088
39: -43.9939919, -10.1884422, -43.9105301, -10.3817406, -22.9918900, 23.0933990
40: -31.3587780, -13.0361176, -31.3182907, -13.0668392, -12.9447975, 12.8863449
41: -19.9167061, 2.0711868, -19.8752155, 2.0214388, -18.7742386, 18.7721863
42: -20.1232147, -3.5145917, -20.0641060, -3.5539136, -13.6077347, 13.5321960

Time for backsubstitution: 2.21 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
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

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4835822, upper bound: 10.4668921
time: 23.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5014215, upper bound: 10.4668921
time: 24.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -29.3725300, -2.2915916, -29.3957424, -2.4675422, -18.0784302, 18.3018379
1: -13.7461758, 2.7292073, -13.7664099, 2.6445777, -11.8266487, 11.9417419
2: -12.0378418, 4.1337500, -12.0653095, 4.0337477, -10.5596466, 10.7012978
3: -21.1479359, -0.7528338, -21.1765308, -0.9253325, -16.1770592, 16.4035950
4: -19.4674759, 2.7805777, -19.5085049, 2.6429100, -14.4107742, 14.6113777
5: -15.5512466, 4.3372974, -15.5792828, 4.1802235, -15.3250732, 15.5313416
6: -21.7200031, -0.4368477, -21.6753407, -0.4500771, -16.7913284, 16.7195816
7: -18.7564240, 2.4813948, -18.7859383, 2.3952565, -16.9301147, 17.0655899
8: -28.8912678, -1.3497291, -28.9469376, -1.4248075, -17.8376770, 17.9878273
9: -19.1533184, 2.6013556, -19.1642666, 2.4145174, -17.7963867, 18.0089340
10: -16.8810940, 5.4347906, -16.8229294, 5.3587518, -19.8771935, 19.8904839
11: -2.8575585, 15.8353558, -2.6538274, 15.8678579, -17.3289413, 17.0838013
12: -17.5107975, 13.1778870, -17.3294525, 13.1961060, -24.5272827, 24.3180847
13: -30.4130135, -1.5539966, -30.3790283, -1.7315750, -20.7970886, 20.9267426
14: -34.2247658, 0.3381319, -34.0560417, 0.3123932, -29.4755478, 29.2752380
15: -15.3598537, 5.2577887, -15.3788214, 5.1341248, -18.5613403, 18.7014656
16: -15.5209179, 6.3498888, -15.4914885, 6.2367907, -19.2502747, 19.3096237
17: -23.1977882, 1.8147719, -23.0277748, 1.8236771, -23.4232941, 23.2001190
18: 1.7088275, 23.2875748, 1.9137712, 23.3114662, -18.9544144, 18.7120552
19: -0.8910236, 11.5627871, -0.7854743, 11.5687876, -11.0991344, 11.0227509
20: -4.5320969, 9.6011915, -4.3970251, 9.5961781, -13.4171829, 13.2916965
21: -1.4682198, 15.6178370, -1.3332615, 15.6276913, -16.1029892, 15.9655647
22: -3.0994737, 11.4947357, -3.0367558, 11.4880352, -13.5696487, 13.5303230
23: -1.4186296, 15.6554832, -1.2871685, 15.6800623, -13.3345146, 13.1808014
24: -1.9528065, 16.3124409, -1.8179016, 16.3414783, -15.1556396, 14.9861412
25: -2.7568250, 16.4192028, -2.6511836, 16.4566383, -17.6094589, 17.4713440
26: -5.5458798, 21.1840172, -5.2827806, 21.2085571, -25.6048203, 25.3044739
27: -0.4980798, 15.6115656, -0.3720016, 15.6049814, -13.5404625, 13.4189682
28: -1.5399847, 15.5091562, -1.4174566, 15.5303240, -14.2274704, 14.1026688
29: -2.0967848, 12.6854172, -2.0057697, 12.6838007, -11.3144722, 11.2066917
30: -8.2365036, 14.8335972, -8.0783615, 14.8769493, -20.2638550, 20.0542831
31: 0.4617434, 16.0317631, 0.5882668, 16.0257015, -14.3255043, 14.2644043
32: -22.0253448, 2.0403967, -22.0347042, 1.9985113, -18.9191589, 18.9838219
33: -39.7388535, -10.3669872, -39.6915932, -10.5214710, -20.9181480, 21.0269012
34: -33.4227676, -10.0187016, -33.3685150, -10.0475817, -17.5300674, 17.5074158
35: -24.0872440, -0.7913582, -24.0649929, -0.8128881, -18.7848816, 18.7661896
36: -20.8179092, 5.2707434, -20.7833195, 5.2599645, -20.1155548, 20.0538483
37: -32.3419189, -2.7112827, -32.2841110, -2.6901436, -26.0819397, 25.9017944
38: -28.8193836, 0.6292605, -28.7475662, 0.6091633, -24.5103683, 24.3833313
39: -43.9954834, -10.1881065, -43.9376717, -10.3745270, -23.0151749, 23.1109314
40: -31.3592358, -13.0359249, -31.3322411, -13.0626898, -12.9612541, 12.8901939
41: -19.9176807, 2.0725005, -19.8860550, 2.0432997, -18.7982330, 18.7888718
42: -20.1238899, -3.5121975, -20.0822506, -3.5379548, -13.6230125, 13.5580025

Time for backsubstitution: 2.22 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1415
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4835822, upper bound: 10.4796790
time: 21.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5014215, upper bound: 10.4796790
time: 20.39 seconds

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

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4810190
time: 21.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4810190
time: 21.74 seconds

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

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4938150
time: 23.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4938150
time: 26.42 seconds

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

Time for backsubstitution: 2.22 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1768

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4941786
time: 27.25 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4941786
time: 18.54 seconds

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

Time for backsubstitution: 2.21 seconds

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
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4891339, upper bound: 10.5069726
time: 24.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.5069733, upper bound: 10.5069726
time: 20.19 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 47.13 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4317206
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4317206
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4445204
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4445204
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4435848
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4993307, upper bound: 10.4435848
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4563868
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4814904, upper bound: 10.4563868
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4869595, upper bound: 10.4575747
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5048110, upper bound: 10.4575747
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4869595, upper bound: 10.4703865
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5048110, upper bound: 10.4703865
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4869595, upper bound: 10.4707150
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5048110, upper bound: 10.4707150
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4869595, upper bound: 10.4835275
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5048110, upper bound: 10.4835275
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4835822, upper bound: 10.4550365
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5014215, upper bound: 10.4550365
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4835822, upper bound: 10.4678220
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5014215, upper bound: 10.4678219
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4835822, upper bound: 10.4668921
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5014215, upper bound: 10.4668921
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4835822, upper bound: 10.4796790
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5014215, upper bound: 10.4796790
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4810190
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4810190
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4938150
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4938150
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4891339, upper bound: 10.4941786
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5069733, upper bound: 10.4941786
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.4891339, upper bound: 10.5069726
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.13
Output dim: 18, lower bound: -10.5069733, upper bound: 10.5069726

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -29.2857895, -2.4889250, -29.2964821, -2.5250316, -17.9510040, 17.9997559
1: -13.7219181, 2.6335177, -13.7196341, 2.6103318, -11.7757874, 11.7978134
2: -12.0016394, 4.0188751, -12.0079975, 3.9963830, -10.4949722, 10.5251217
3: -21.0633583, -0.9454036, -21.0906219, -0.9711900, -16.0658722, 16.1203728
4: -19.3963528, 2.6149492, -19.4017944, 2.5885644, -14.3034286, 14.3349228
5: -15.4722080, 4.1594706, -15.4928141, 4.1313591, -15.2132530, 15.2634697
6: -21.6621838, -0.4756622, -21.6564331, -0.4731302, -16.6675797, 16.6540680
7: -18.6944695, 2.3747134, -18.7045479, 2.3525219, -16.8367233, 16.8719788
8: -28.8359032, -1.4647932, -28.8433266, -1.4829059, -17.7427597, 17.7676964
9: -19.0509186, 2.3977950, -19.0644302, 2.3654385, -17.6627579, 17.7082596
10: -16.7940960, 5.3235998, -16.7722549, 5.3195944, -19.7431717, 19.7237167
11: -2.6113927, 15.7401571, -2.5871563, 15.7570105, -16.9882202, 16.9462013
12: -17.3101616, 13.0420742, -17.2903900, 13.0813847, -24.2071152, 24.1499786
13: -30.3527908, -1.7126279, -30.3362122, -1.7749729, -20.6886978, 20.7354355
14: -34.0020103, 0.2192950, -33.9883537, 0.2452455, -29.1541290, 29.1117325
15: -15.2921944, 5.1065931, -15.3047876, 5.0881681, -18.4622803, 18.4914932
16: -15.4452953, 6.2451353, -15.4423628, 6.2149239, -19.1269226, 19.1555099
17: -23.0123177, 1.6733794, -22.9936180, 1.7079806, -23.0936813, 23.0331345
18: 1.9447641, 23.1985054, 1.9841037, 23.2126236, -18.6266823, 18.5748787
19: -0.7667165, 11.5504971, -0.7472608, 11.5504313, -10.9649506, 10.9458103
20: -4.3829422, 9.5702705, -4.3572140, 9.5765648, -13.2506294, 13.2192345
21: -1.3033113, 15.5787935, -1.2827525, 15.5889902, -15.9128113, 15.8852882
22: -3.0310876, 11.4722462, -3.0104930, 11.4709625, -13.4863968, 13.4651260
23: -1.2664900, 15.6328907, -1.2348175, 15.6247463, -13.1293297, 13.1066494
24: -1.7926955, 16.2668934, -1.7585115, 16.2679462, -14.9260330, 14.8936119
25: -2.6217780, 16.3999348, -2.5940170, 16.4032135, -17.4243698, 17.3954582
26: -5.2519298, 21.0962601, -5.2056851, 21.1194477, -25.2144547, 25.1461792
27: -0.3624797, 15.5574198, -0.3362842, 15.5648689, -13.3732567, 13.3414917
28: -1.3951530, 15.4848099, -1.3677392, 15.4803724, -14.0399246, 14.0176353
29: -2.0021136, 12.6432495, -1.9812329, 12.6459913, -11.1725159, 11.1494751
30: -8.0360527, 14.7605743, -8.0062180, 14.7653971, -19.9581299, 19.9251251
31: 0.5986929, 16.0109138, 0.6250572, 16.0114059, -14.2030907, 14.1778526
32: -21.9762878, 1.9644175, -21.9780350, 1.9555440, -18.8316879, 18.8396301
33: -39.6343460, -10.5286980, -39.6356773, -10.5547066, -20.7859802, 20.8144455
34: -33.3169937, -10.1016378, -33.3060303, -10.1094723, -17.3674698, 17.3606949
35: -24.0075016, -0.8609681, -24.0087547, -0.8805318, -18.6298370, 18.6460228
36: -20.7595844, 5.2371826, -20.7481499, 5.2144337, -19.9883499, 19.9933014
37: -32.2170563, -2.7482252, -32.2095528, -2.7661800, -25.8109436, 25.8028870
38: -28.7164822, 0.5751796, -28.7057076, 0.5622478, -24.2950897, 24.2872391
39: -43.8914642, -10.3417492, -43.8715897, -10.3919964, -22.8909760, 22.9146194
40: -31.3066216, -13.0636044, -31.3021278, -13.0741186, -12.8501358, 12.8451996
41: -19.8581238, 2.0139089, -19.8610935, 2.0087237, -18.6947556, 18.7004852
42: -20.0605736, -3.5668280, -20.0515385, -3.5661242, -13.4855309, 13.4729385

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1768
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
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 973
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
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4762408, upper bound: 10.4095719
time: 22.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4807995, upper bound: 10.4310257
time: 21.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -29.3317680, -2.4571795, -29.2967911, -2.5107574, -18.0113983, 18.0237541
1: -13.7520285, 2.6548131, -13.7200737, 2.6200471, -11.8163681, 11.8151550
2: -12.0325966, 4.0396204, -12.0080404, 4.0055141, -10.5352020, 10.5414219
3: -21.0967579, -0.9217629, -21.0908432, -0.9615359, -16.1092377, 16.1404266
4: -19.4567757, 2.6466384, -19.4022732, 2.6034985, -14.3789902, 14.3586349
5: -15.5087271, 4.1848869, -15.4930468, 4.1424046, -15.2611656, 15.2845078
6: -21.6662655, -0.4626951, -21.6568222, -0.4718895, -16.6733017, 16.6699219
7: -18.7328262, 2.3966107, -18.7050648, 2.3623247, -16.8868332, 16.8907852
8: -28.9018211, -1.4275732, -28.8437138, -1.4649577, -17.8262024, 17.7962074
9: -19.0899181, 2.4229193, -19.0649548, 2.3764997, -17.7126808, 17.7280846
10: -16.8243942, 5.3444757, -16.7737312, 5.3272247, -19.7804642, 19.7456131
11: -2.6472771, 15.7920046, -2.6018124, 15.7573719, -17.0184212, 17.0119286
12: -17.3264198, 13.0806713, -17.2965393, 13.0827503, -24.2239914, 24.1949844
13: -30.3672886, -1.6957536, -30.3373718, -1.7720599, -20.6987991, 20.7612572
14: -34.0259247, 0.2313061, -33.9918976, 0.2484350, -29.1797333, 29.1349869
15: -15.3314228, 5.1306677, -15.3056307, 5.0980654, -18.5111008, 18.5132256
16: -15.4601154, 6.2592659, -15.4436131, 6.2175784, -19.1365242, 19.1859856
17: -23.0278091, 1.6999743, -22.9963017, 1.7089357, -23.1080017, 23.0755539
18: 1.9057064, 23.2431297, 1.9685311, 23.2128448, -18.6621323, 18.6353226
19: -0.7857742, 11.5645924, -0.7543697, 11.5505581, -10.9828835, 10.9672012
20: -4.4002333, 9.5799608, -4.3614016, 9.5775318, -13.2695122, 13.2368202
21: -1.3290043, 15.6001167, -1.2924843, 15.5893116, -15.9349213, 15.9146767
22: -3.0435848, 11.4771137, -3.0140431, 11.4713230, -13.5007095, 13.4738083
23: -1.2974691, 15.6719780, -1.2485566, 15.6250782, -13.1553726, 13.1597672
24: -1.8265300, 16.3089333, -1.7718139, 16.2680893, -14.9570923, 14.9492378
25: -2.6518993, 16.4350719, -2.6048827, 16.4033813, -17.4529266, 17.4409904
26: -5.2934618, 21.1328812, -5.2217731, 21.1196117, -25.2549820, 25.1995926
27: -0.3775516, 15.5742531, -0.3405223, 15.5653400, -13.3864670, 13.3610916
28: -1.4242005, 15.5238428, -1.3798242, 15.4808693, -14.0668564, 14.0684357
29: -2.0148878, 12.6578560, -1.9863613, 12.6463470, -11.1835823, 11.1710396
30: -8.0774326, 14.8232403, -8.0227423, 14.7656431, -19.9948120, 20.0036697
31: 0.5823903, 16.0177193, 0.6196303, 16.0120029, -14.2176094, 14.1851997
32: -22.0043221, 1.9872837, -21.9787159, 1.9616361, -18.8758774, 18.8610382
33: -39.6499481, -10.5154600, -39.6404266, -10.5536613, -20.8025436, 20.8316574
34: -33.3398476, -10.0639019, -33.3140182, -10.1087589, -17.3867531, 17.4070702
35: -24.0368538, -0.8171539, -24.0195446, -0.8795288, -18.6554413, 18.7007904
36: -20.7761078, 5.2666407, -20.7545013, 5.2148561, -20.0041046, 20.0292358
37: -32.2503433, -2.7033386, -32.2215691, -2.7657547, -25.8394547, 25.8600464
38: -28.7301350, 0.5946689, -28.7082596, 0.5633240, -24.3114853, 24.3186798
39: -43.9136620, -10.3362169, -43.8733292, -10.3909302, -22.9025879, 22.9300690
40: -31.3134575, -13.0608807, -31.3026276, -13.0743723, -12.8520317, 12.8557701
41: -19.8634911, 2.0295053, -19.8623581, 2.0105522, -18.7058487, 18.7161331
42: -20.0694122, -3.5541179, -20.0525551, -3.5637796, -13.5057030, 13.4877510

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1768
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
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 973
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
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1583
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4940829, upper bound: 10.4095719
time: 25.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4986398, upper bound: 10.4310257
time: 30.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -29.2859669, -2.4781790, -29.3522949, -2.4934397, -17.9725876, 18.0634956
1: -13.7222843, 2.6401269, -13.7556562, 2.6294205, -11.7896347, 11.8396645
2: -12.0016127, 4.0266743, -12.0478706, 4.0184064, -10.5102768, 10.5719910
3: -21.0634480, -0.9376502, -21.1324577, -0.9447861, -16.0881195, 16.1700401
4: -19.3968430, 2.6257133, -19.4736462, 2.6180148, -14.3234940, 14.4175072
5: -15.4723139, 4.1686096, -15.5393877, 4.1593528, -15.2366486, 15.3193550
6: -21.6624279, -0.4747543, -21.6672840, -0.4594059, -16.6827927, 16.6668549
7: -18.6948051, 2.3838453, -18.7569580, 2.3775001, -16.8564911, 16.9342194
8: -28.8363037, -1.4514847, -28.9221821, -1.4490213, -17.7652206, 17.8586807
9: -19.0514526, 2.4063172, -19.1148262, 2.3925126, -17.6828384, 17.7667656
10: -16.7952042, 5.3280721, -16.8066406, 5.3408775, -19.7651978, 19.7647667
11: -2.6211684, 15.7404070, -2.6218996, 15.8185654, -17.0566483, 16.9741173
12: -17.3164864, 13.0431271, -17.3103180, 13.1335173, -24.2679138, 24.1665421
13: -30.3539276, -1.7108216, -30.3544712, -1.7493973, -20.7268448, 20.7495689
14: -34.0048561, 0.2204790, -34.0255051, 0.2611232, -29.1901398, 29.1393127
15: -15.2928801, 5.1122117, -15.3494387, 5.1110449, -18.4814987, 18.5395584
16: -15.4464388, 6.2468414, -15.4689627, 6.2279201, -19.1593437, 19.1725311
17: -23.0145760, 1.6740816, -23.0128670, 1.7575483, -23.1646729, 23.0457687
18: 1.9325867, 23.1987991, 1.9432821, 23.2709370, -18.6974487, 18.6121063
19: -0.7706518, 11.5505733, -0.7680798, 11.5654631, -10.9842339, 10.9662209
20: -4.3854198, 9.5704527, -4.3766670, 9.5837126, -13.2606850, 13.2398911
21: -1.3089790, 15.5790043, -1.3079095, 15.6109600, -15.9365730, 15.9077072
22: -3.0328834, 11.4725380, -3.0239372, 11.4795036, -13.4940071, 13.4850006
23: -1.2758303, 15.6330986, -1.2637358, 15.6696215, -13.1837349, 13.1317120
24: -1.8032537, 16.2669868, -1.7954798, 16.3220520, -14.9910965, 14.9277878
25: -2.6304445, 16.4000797, -2.6277909, 16.4491940, -17.4788666, 17.4274254
26: -5.2650747, 21.0964870, -5.2504826, 21.1706619, -25.2797241, 25.1910782
27: -0.3644457, 15.5578222, -0.3543258, 15.5805321, -13.3906784, 13.3580933
28: -1.4024968, 15.4851580, -1.3940434, 15.5203829, -14.0870323, 14.0422020
29: -2.0060868, 12.6434813, -1.9943962, 12.6646099, -11.1970634, 11.1595230
30: -8.0478764, 14.7607174, -8.0461006, 14.8395672, -20.0424423, 19.9572144
31: 0.5963035, 16.0112000, 0.6080852, 16.0179043, -14.2046356, 14.2002296
32: -21.9767952, 1.9702764, -22.0201893, 1.9832115, -18.8551788, 18.8999519
33: -39.6378784, -10.5280762, -39.6557159, -10.5352764, -20.8068619, 20.8340759
34: -33.3244591, -10.1010437, -33.3340187, -10.0564175, -17.4284363, 17.3831406
35: -24.0166111, -0.8602831, -24.0418797, -0.8221211, -18.6973572, 18.6743546
36: -20.7649078, 5.2373953, -20.7681789, 5.2516670, -20.0322800, 20.0119247
37: -32.2282867, -2.7480011, -32.2520599, -2.6996517, -25.8890686, 25.8396454
38: -28.7204781, 0.5760255, -28.7298813, 0.5951862, -24.3370056, 24.3116226
39: -43.8929863, -10.3414135, -43.8987198, -10.3847876, -22.9142609, 22.9321556
40: -31.3070889, -13.0633688, -31.3161030, -13.0699139, -12.8665657, 12.8490410
41: -19.8591042, 2.0152562, -19.8719444, 2.0305307, -18.7186966, 18.7171707
42: -20.0612049, -3.5644631, -20.0696983, -3.5501485, -13.5007477, 13.4987106

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1768
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
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 873
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
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1567
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 981
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4762408, upper bound: 10.4223647
time: 27.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4807995, upper bound: 10.4438251
time: 22.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -29.3319969, -2.4464664, -29.3526764, -2.4791641, -18.0329781, 18.0871162
1: -13.7524214, 2.6614125, -13.7560577, 2.6391661, -11.8302231, 11.8567810
2: -12.0325994, 4.0474138, -12.0479317, 4.0275517, -10.5505028, 10.5883064
3: -21.0968513, -0.9140315, -21.1326427, -0.9350839, -16.1315231, 16.1900711
4: -19.4572830, 2.6574602, -19.4741344, 2.6329956, -14.3990479, 14.4411812
5: -15.5088301, 4.1940460, -15.5396576, 4.1703930, -15.2845154, 15.3404083
6: -21.6664944, -0.4617519, -21.6676197, -0.4581776, -16.6884384, 16.6827850
7: -18.7331429, 2.4057436, -18.7574806, 2.3873277, -16.9066620, 16.9530182
8: -28.9021530, -1.4142647, -28.9225750, -1.4311342, -17.8486862, 17.8871613
9: -19.0904465, 2.4314265, -19.1153431, 2.4035707, -17.7326889, 17.7865982
10: -16.8254814, 5.3489819, -16.8080807, 5.3485193, -19.8024635, 19.7866554
11: -2.6570482, 15.7922401, -2.6365480, 15.8189545, -17.0868721, 17.0398178
12: -17.3327866, 13.0818510, -17.3164940, 13.1348438, -24.2845612, 24.2115173
13: -30.3683510, -1.6940236, -30.3556519, -1.7465816, -20.7369156, 20.7753029
14: -34.0287552, 0.2325215, -34.0290756, 0.2643499, -29.2154846, 29.1624069
15: -15.3321238, 5.1362286, -15.3502693, 5.1209545, -18.5303116, 18.5613518
16: -15.4612465, 6.2609596, -15.4702415, 6.2305527, -19.1687927, 19.2029305
17: -23.0300694, 1.7006867, -23.0156136, 1.7585287, -23.1789627, 23.0881348
18: 1.8935347, 23.2434406, 1.9277077, 23.2711334, -18.7329102, 18.6725273
19: -0.7897053, 11.5646706, -0.7752018, 11.5655861, -11.0021629, 10.9876308
20: -4.4026890, 9.5801601, -4.3808422, 9.5846920, -13.2795868, 13.2574692
21: -1.3346634, 15.6003475, -1.3176289, 15.6112747, -15.9586258, 15.9371109
22: -3.0453858, 11.4774065, -3.0274086, 11.4798746, -13.5083046, 13.4936218
23: -1.3067837, 15.6721821, -1.2774544, 15.6699295, -13.2097855, 13.1848145
24: -1.8370585, 16.3090324, -1.8087816, 16.3221912, -15.0221558, 14.9834251
25: -2.6605644, 16.4352036, -2.6386676, 16.4493809, -17.5074692, 17.4729729
26: -5.3066287, 21.1331100, -5.2665510, 21.1708794, -25.3202438, 25.2445068
27: -0.3795056, 15.5746641, -0.3585687, 15.5809803, -13.4039001, 13.3776779
28: -1.4315495, 15.5241947, -1.4061060, 15.5208874, -14.1139488, 14.0929794
29: -2.0188725, 12.6580696, -1.9995260, 12.6649437, -11.2079582, 11.1810837
30: -8.0892344, 14.8234339, -8.0626068, 14.8398018, -20.0791168, 20.0357437
31: 0.5799985, 16.0179939, 0.6026282, 16.0185127, -14.2192230, 14.2074203
32: -22.0047607, 1.9930820, -22.0208378, 1.9893227, -18.8993530, 18.9213562
33: -39.6534882, -10.5147572, -39.6604462, -10.5342531, -20.8234100, 20.8513107
34: -33.3473129, -10.0632954, -33.3420334, -10.0556917, -17.4477234, 17.4294968
35: -24.0459709, -0.8164759, -24.0526390, -0.8211365, -18.7229462, 18.7291603
36: -20.7814064, 5.2668839, -20.7745743, 5.2520328, -20.0480576, 20.0478592
37: -32.2615089, -2.7031016, -32.2641411, -2.6991777, -25.9175720, 25.8968277
38: -28.7341576, 0.5955582, -28.7324715, 0.5962868, -24.3534698, 24.3430710
39: -43.9151001, -10.3358850, -43.9004440, -10.3836946, -22.9259033, 22.9476509
40: -31.3139153, -13.0606050, -31.3165455, -13.0701714, -12.8684540, 12.8595810
41: -19.8644943, 2.0308414, -19.8732262, 2.0323567, -18.7297440, 18.7328339
42: -20.0700893, -3.5517292, -20.0706863, -3.5477810, -13.5209274, 13.5135269

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1768
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
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 873
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
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1567
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 981
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4940829, upper bound: 10.4223647
time: 20.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -10.4986398, upper bound: 10.4438251
time: 19.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -29.3143387, -2.4180412, -29.3108215, -2.5236702, -17.9738350, 18.0862160
1: -13.7338800, 2.6726029, -13.7254524, 2.6113138, -11.7851067, 11.8434219
2: -12.0165310, 4.0655975, -12.0156403, 3.9972596, -10.5079880, 10.5798187
3: -21.0892372, -0.8721409, -21.1035461, -0.9696689, -16.0875549, 16.2085876
4: -19.4256153, 2.6746221, -19.4157696, 2.5896306, -14.3262253, 14.4071159
5: -15.4947224, 4.2253366, -15.5038977, 4.1325541, -15.2324066, 15.3416252
6: -21.6869717, -0.4649773, -21.6576633, -0.4702034, -16.6993713, 16.6646996
7: -18.7176037, 2.4210324, -18.7161713, 2.3532748, -16.8559875, 16.9303665
8: -28.8625698, -1.4138918, -28.8559055, -1.4819231, -17.7641983, 17.8313828
9: -19.0875034, 2.4707317, -19.0814972, 2.3666096, -17.6941719, 17.7983246
10: -16.8349361, 5.3534813, -16.7742958, 5.3208084, -19.7911911, 19.7523308
11: -2.6815321, 15.7731314, -2.5889089, 15.7722425, -17.0691338, 16.9725227
12: -17.3848934, 13.0716095, -17.2925301, 13.0939484, -24.2947845, 24.1810760
13: -30.3767815, -1.6512551, -30.3465729, -1.7727494, -20.7082367, 20.8083572
14: -34.0434723, 0.2403307, -33.9936790, 0.2461810, -29.2060852, 29.1371918
15: -15.3080301, 5.1523328, -15.3099518, 5.0902958, -18.4799957, 18.5421066
16: -15.4755077, 6.3035450, -15.4526882, 6.2155533, -19.1538544, 19.2240257
17: -23.0526352, 1.6957505, -22.9954720, 1.7110963, -23.1393280, 23.0561142
18: 1.8603792, 23.2246094, 1.9814219, 23.2215424, -18.7207794, 18.5987816
19: -0.8072989, 11.5554609, -0.7487185, 11.5506458, -11.0034847, 10.9574490
20: -4.4281292, 9.5751476, -4.3600626, 9.5767689, -13.2948456, 13.2319717
21: -1.3393760, 15.5865536, -1.2852564, 15.5895777, -15.9479752, 15.9009438
22: -3.0591817, 11.4756203, -3.0117292, 11.4723129, -13.5167198, 13.4764786
23: -1.3204956, 15.6504526, -1.2363777, 15.6332521, -13.1872368, 13.1188431
24: -1.8512177, 16.2897644, -1.7594485, 16.2765598, -14.9913635, 14.9123535
25: -2.6728024, 16.4084167, -2.5957260, 16.4051781, -17.4735947, 17.4066429
26: -5.3445978, 21.1104641, -5.2079282, 21.1213150, -25.3135605, 25.1644211
27: -0.4108238, 15.5722370, -0.3380089, 15.5708141, -13.4255447, 13.3540115
28: -1.4401598, 15.4940701, -1.3695488, 15.4829998, -14.0846519, 14.0313377
29: -2.0350246, 12.6614552, -1.9817750, 12.6540947, -11.2153549, 11.1635780
30: -8.0843697, 14.7801809, -8.0080595, 14.7738667, -20.0118790, 19.9434814
31: 0.5483770, 16.0162106, 0.6227837, 16.0124702, -14.2528839, 14.1926956
32: -21.9937973, 1.9676132, -21.9779186, 1.9561858, -18.8552704, 18.8440285
33: -39.6525269, -10.4821806, -39.6332092, -10.5529423, -20.8141708, 20.8549881
34: -33.3497620, -10.0934849, -33.3074875, -10.1081123, -17.4071922, 17.3731041
35: -24.0193558, -0.8348722, -24.0060368, -0.8785472, -18.6493835, 18.6704102
36: -20.7772865, 5.2413559, -20.7493248, 5.2152357, -20.0128098, 19.9979706
37: -32.2537270, -2.7485461, -32.2118111, -2.7689495, -25.8614349, 25.8118057
38: -28.7701855, 0.5873938, -28.7089062, 0.5661535, -24.3693924, 24.3011551
39: -43.9266472, -10.2996855, -43.8787384, -10.3912106, -22.9268951, 22.9578323
40: -31.3273659, -13.0598669, -31.3042679, -13.0738411, -12.8771896, 12.8502235
41: -19.8883610, 2.0262978, -19.8622818, 2.0119779, -18.7324219, 18.7147064
42: -20.0927410, -3.5578318, -20.0527878, -3.5646944, -13.5272408, 13.4828987

Time for backsubstitution: 2.18 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 34.39 + 1767.67 = 1802.06 seconds
