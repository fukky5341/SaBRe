## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 1800 seconds
Split limit: 100
Threshold: 12.2525096256


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318)
1: (-9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788)
2: (-8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7231445, 20.7231445)
3: (-9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082)
4: (-11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1413574, 25.1413574)
5: (-9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5836639, 23.5836563)
6: (-20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582)
7: (-11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5392303, 27.5392303)
8: (-13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.7004547, 30.7004623)
9: (-6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373)
10: (-15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645)
11: (-17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980)
12: (-22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2585602, 30.2585526)
13: (-16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340)
14: (-35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908)
15: (-14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580)
16: (-17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968)
17: (-38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160)
18: (-19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279)
19: (-15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895)
20: (-11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985)
21: (-17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173)
22: (-20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003)
23: (-14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201)
24: (-17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850)
25: (-14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606)
26: (-21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424)
27: (-17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042)
28: (-14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075)
29: (-21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177)
30: (-16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568)
31: (-19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244)
32: (-19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185)
33: (-33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8120346, 37.8120346)
34: (-31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6454086, 29.6454086)
35: (-30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6890564, 30.6890564)
36: (-27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2380524, 31.2380600)
37: (-39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6004257, 36.6004257)
38: (-32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296)
39: (-37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3137512, 42.3137512)
40: (-30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616)
41: (-21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2781067, 27.2781067)
42: (-12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.89 + 55.99 = 58.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -12.2647744, upper bound: 12.2647744

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1705

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 666

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2627719, upper bound: 12.2413949
time: 44.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2627719, upper bound: 12.2631999
time: 36.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 81.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 81.29
Output dim: 9, lower bound: -12.2627719, upper bound: 12.2413949
IS_A2, status: Status.UNKNOWN, split count: 1, time: 81.29
Output dim: 9, lower bound: -12.2627719, upper bound: 12.2631999

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -22.5981331, 10.1343727, -22.6166916, 10.1441975, -32.7423325, 32.7510643
1: -9.0543194, 13.9815416, -9.0646629, 13.9892578, -23.0435772, 23.0462036
2: -8.0801525, 12.9094820, -8.0884008, 12.9143906, -20.7092972, 20.7139473
3: -9.3911724, 14.5746975, -9.4041634, 14.5879202, -23.9790916, 23.9788609
4: -11.1043615, 13.9946880, -11.1280661, 14.0188122, -25.1020203, 25.1015396
5: -9.1662207, 14.6412067, -9.1734009, 14.6505451, -23.5674896, 23.5648422
6: -20.4720421, 7.4209018, -20.5006618, 7.4550519, -27.9270935, 27.9215641
7: -11.2851448, 16.7832870, -11.2939148, 16.7919559, -27.5240555, 27.5234604
8: -13.3560085, 17.4009686, -13.3772068, 17.4217262, -30.6653824, 30.6673508
9: -6.9595094, 16.0779934, -6.9932122, 16.1114597, -23.0709686, 23.0712051
10: -15.3183756, 19.5921783, -15.3307228, 19.6095314, -34.9279060, 34.9229012
11: -17.7680283, 12.7553988, -17.7918854, 12.7789726, -30.5470009, 30.5472832
12: -22.0003967, 9.5520725, -22.0485954, 9.6008854, -30.1816635, 30.1811371
13: -16.9403667, 14.1636600, -16.9554539, 14.1845322, -31.1248989, 31.1191139
14: -35.5948029, 5.6657543, -35.6363602, 5.7040873, -41.2988892, 41.3021164
15: -14.0737524, 10.5149946, -14.0946598, 10.5215607, -24.5953140, 24.6096535
16: -17.5774651, 14.2200222, -17.6001186, 14.2437344, -31.8211994, 31.8201408
17: -38.8902359, 10.3410025, -38.9552155, 10.4070406, -49.2972755, 49.2962189
18: -19.2736626, 7.6653910, -19.2915878, 7.6759195, -26.9495811, 26.9569778
19: -15.6452141, 3.5614948, -15.6458130, 3.5672505, -19.2124653, 19.2073078
20: -11.3735209, 7.3634405, -11.3957958, 7.3828702, -18.7563915, 18.7592354
21: -17.7072029, 6.7362986, -17.7191887, 6.7478447, -24.4550476, 24.4554863
22: -20.7304554, 6.4394264, -20.7641068, 6.4714842, -27.2019386, 27.2035332
23: -14.2639656, 5.9744225, -14.2776718, 5.9849892, -20.2489548, 20.2520943
24: -17.5106888, 7.5322170, -17.5242081, 7.5377378, -25.0484276, 25.0564251
25: -14.8123493, 7.4330845, -14.8304062, 7.4525561, -22.2649059, 22.2634907
26: -21.2917118, 10.0945625, -21.3230553, 10.1140165, -31.4057274, 31.4176178
27: -17.5334187, 8.2798643, -17.5521660, 8.2875423, -25.8209610, 25.8320312
28: -14.3788509, 7.1330004, -14.3875656, 7.1394210, -21.5182724, 21.5205650
29: -21.8448830, 8.5984669, -21.8974113, 8.6480484, -30.4929314, 30.4958782
30: -16.5225544, 9.8081408, -16.5330811, 9.8172626, -26.3398170, 26.3412209
31: -19.2464523, 5.6937418, -19.2564945, 5.7056088, -24.9520607, 24.9502373
32: -19.1461601, 8.2213621, -19.1748638, 8.2487259, -27.3948860, 27.3962250
33: -33.5403519, 4.6151047, -33.5909653, 4.6589642, -37.7328491, 37.7393570
34: -31.4704647, -0.9309711, -31.5086136, -0.8991833, -29.5869598, 29.5949402
35: -30.2922745, 1.2496948, -30.3291607, 1.2787342, -30.6324844, 30.6395187
36: -27.1588821, 4.0920048, -27.1749935, 4.1055431, -31.2144547, 31.2155838
37: -38.9903030, -1.9979553, -39.0237579, -1.9795837, -36.5367432, 36.5596619
38: -32.2113190, 3.8182402, -32.2320786, 3.8348160, -36.0461349, 36.0503197
39: -37.8289528, 4.4278870, -37.8755112, 4.4658947, -42.2412720, 42.2495728
40: -30.2279358, 4.3797865, -30.2651138, 4.4054050, -34.6333389, 34.6449013
41: -21.4516983, 5.8731050, -21.4633408, 5.8822460, -27.2596283, 27.2619858
42: -12.5312729, 7.1041317, -12.5546341, 7.1293006, -19.6605740, 19.6587658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2595753, upper bound: 12.2080936
time: 54.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2614160, upper bound: 12.2400381
time: 53.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.6201763, 10.1463680, -22.6211853, 10.1468182, -32.7669945, 32.7675552
1: -9.0670042, 13.9911346, -9.0675049, 13.9916592, -23.0586624, 23.0586395
2: -8.0899296, 12.9154358, -8.0904074, 12.9159603, -20.7224884, 20.7221527
3: -9.4053745, 14.5930882, -9.4057388, 14.5941811, -23.9995556, 23.9988270
4: -11.1298599, 14.0302086, -11.1306438, 14.0312843, -25.1399460, 25.1347351
5: -9.1735439, 14.6534481, -9.1756296, 14.6543293, -23.5794296, 23.5817719
6: -20.5131073, 7.4578886, -20.5144615, 7.4588361, -27.9719429, 27.9723511
7: -11.2967358, 16.7930851, -11.2972050, 16.7940712, -27.5379868, 27.5374527
8: -13.3787842, 17.4296856, -13.3792524, 17.4305439, -30.7020187, 30.6987305
9: -6.9938941, 16.1277561, -6.9955778, 16.1297741, -23.1236687, 23.1233330
10: -15.3327293, 19.6110630, -15.3343115, 19.6157036, -34.9484329, 34.9453735
11: -17.8016014, 12.7790470, -17.8029671, 12.7803965, -30.5819969, 30.5820141
12: -22.0704346, 9.6046276, -22.0721436, 9.6053362, -30.2429810, 30.2569504
13: -16.9590282, 14.1873074, -16.9607048, 14.1879311, -31.1469593, 31.1480122
14: -35.6544685, 5.7049923, -35.6559334, 5.7058220, -41.3602905, 41.3609238
15: -14.0968027, 10.5234375, -14.0982933, 10.5244312, -24.6212349, 24.6217308
16: -17.6042233, 14.2404671, -17.6066170, 14.2517529, -31.8559761, 31.8470840
17: -38.9856262, 10.4077120, -38.9877472, 10.4096336, -49.3952599, 49.3954582
18: -19.2963181, 7.6755295, -19.2986984, 7.6770868, -26.9734039, 26.9742279
19: -15.6481133, 3.5668759, -15.6490765, 3.5679221, -19.2160358, 19.2159519
20: -11.4046688, 7.3833785, -11.4055986, 7.3841252, -18.7887936, 18.7889767
21: -17.7235317, 6.7480869, -17.7244949, 6.7490082, -24.4725399, 24.4725819
22: -20.7744389, 6.4723949, -20.7781811, 6.4733009, -27.2477398, 27.2505760
23: -14.2826786, 5.9854822, -14.2835169, 5.9863386, -20.2690163, 20.2689991
24: -17.5281887, 7.5386620, -17.5292473, 7.5391655, -25.0673542, 25.0679092
25: -14.8372412, 7.4537659, -14.8388548, 7.4543958, -22.2916374, 22.2926216
26: -21.3344841, 10.1134291, -21.3365536, 10.1149101, -31.4493942, 31.4499817
27: -17.5574379, 8.2880526, -17.5593948, 8.2889862, -25.8464241, 25.8474464
28: -14.3899946, 7.1393294, -14.3907061, 7.1402645, -21.5302582, 21.5300350
29: -21.9206409, 8.6490402, -21.9223747, 8.6495018, -30.5701427, 30.5714149
30: -16.5362949, 9.8184347, -16.5371418, 9.8194027, -26.3556976, 26.3555756
31: -19.2587528, 5.7062082, -19.2598190, 5.7069931, -24.9657459, 24.9660263
32: -19.1869545, 8.2509460, -19.1882000, 8.2517042, -27.4386597, 27.4391460
33: -33.5951309, 4.6795464, -33.5964279, 4.6811218, -37.8096771, 37.7993317
34: -31.5120029, -0.8859615, -31.5126228, -0.8841610, -29.6438599, 29.6327515
35: -30.3325100, 1.2922421, -30.3333664, 1.2934780, -30.6874161, 30.6865921
36: -27.1783428, 4.1068630, -27.1808643, 4.1076145, -31.2316360, 31.2359009
37: -39.0285568, -1.9753494, -39.0302658, -1.9719162, -36.6138763, 36.5937958
38: -32.2347107, 3.8385582, -32.2399178, 3.8401356, -36.0748444, 36.0784760
39: -37.8777618, 4.4840202, -37.8799744, 4.4860382, -42.3101196, 42.3065643
40: -30.2690353, 4.4166279, -30.2703724, 4.4180489, -34.6870842, 34.6870003
41: -21.4656391, 5.8843269, -21.4665756, 5.8850355, -27.2764130, 27.2767868
42: -12.5642853, 7.1310949, -12.5653248, 7.1318016, -19.6960869, 19.6964188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2595753, upper bound: 12.2080936
time: 62.23 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2618522, upper bound: 12.2618512
time: 40.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 105.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 105.57
Output dim: 9, lower bound: -12.2595753, upper bound: 12.2080936
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 105.57
Output dim: 9, lower bound: -12.2614160, upper bound: 12.2400381
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 105.57
Output dim: 9, lower bound: -12.2595753, upper bound: 12.2080936
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 105.57
Output dim: 9, lower bound: -12.2618522, upper bound: 12.2618512

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -22.5907516, 10.1224403, -22.5956478, 10.1090355, -32.6997871, 32.7180862
1: -9.0503674, 13.9749813, -9.0517139, 13.9707355, -23.0211029, 23.0266953
2: -8.0759945, 12.9021730, -8.0769901, 12.8938723, -20.6844177, 20.6957474
3: -9.3798313, 14.5710106, -9.3684149, 14.5761375, -23.9559689, 23.9394264
4: -11.0974369, 13.9816542, -11.1103992, 13.9787350, -25.0529175, 25.0666275
5: -9.1610632, 14.6362572, -9.1553726, 14.6329117, -23.5441055, 23.5413818
6: -20.4460526, 7.4163661, -20.4244175, 7.4208312, -27.8668842, 27.8407841
7: -11.2798557, 16.7789650, -11.2767525, 16.7793522, -27.5053024, 27.5011444
8: -13.3515835, 17.3916130, -13.3637037, 17.3956509, -30.6349792, 30.6445694
9: -6.9353371, 16.0749969, -6.9215755, 16.0797043, -23.0150414, 22.9965725
10: -15.3060617, 19.5860291, -15.2937279, 19.5887928, -34.8948555, 34.8797569
11: -17.7591896, 12.7389183, -17.7565861, 12.7309542, -30.4901428, 30.4955044
12: -21.9780388, 9.5447712, -21.9837742, 9.5696802, -30.1149139, 30.1035080
13: -16.9050617, 14.1574593, -16.8540974, 14.1320391, -31.0371017, 31.0115566
14: -35.5850754, 5.6247683, -35.5775642, 5.5896425, -41.1747169, 41.2023315
15: -14.0688343, 10.5053768, -14.0779152, 10.4927416, -24.5615768, 24.5832920
16: -17.5665665, 14.2147131, -17.5630760, 14.2282362, -31.7948036, 31.7777901
17: -38.8815918, 10.2917233, -38.8895416, 10.2666435, -49.1482353, 49.1812668
18: -19.2676506, 7.6192517, -19.2325859, 7.5438037, -26.8114548, 26.8518372
19: -15.6406250, 3.5430367, -15.6162710, 3.5138068, -19.1544323, 19.1593075
20: -11.3681278, 7.3521481, -11.3727436, 7.3515253, -18.7196541, 18.7248917
21: -17.7002907, 6.7166958, -17.6862221, 6.6942053, -24.3944969, 24.4029179
22: -20.7240429, 6.4146166, -20.7238140, 6.3990660, -27.1231079, 27.1384315
23: -14.2589464, 5.9540467, -14.2472000, 5.9278817, -20.1868286, 20.2012463
24: -17.5034866, 7.4976501, -17.4715691, 7.4373522, -24.9408379, 24.9692192
25: -14.8067703, 7.4092999, -14.7963762, 7.3844967, -22.1912670, 22.2056770
26: -21.2831364, 10.0617409, -21.2687321, 10.0195389, -31.3026752, 31.3304729
27: -17.5262985, 8.2489471, -17.5012703, 8.2011843, -25.7274818, 25.7502174
28: -14.3736153, 7.1145458, -14.3593645, 7.0862265, -21.4598427, 21.4739113
29: -21.8388443, 8.5747623, -21.8666382, 8.5802956, -30.4191399, 30.4414005
30: -16.5154152, 9.7859411, -16.4981747, 9.7549419, -26.2703571, 26.2841148
31: -19.2394714, 5.6647120, -19.2116241, 5.6233845, -24.8628559, 24.8763351
32: -19.1123657, 8.2156887, -19.0773029, 8.2055264, -27.3178921, 27.2929916
33: -33.5186920, 4.6113186, -33.5266418, 4.6358318, -37.6842041, 37.6683121
34: -31.4591427, -0.9339371, -31.4754448, -0.9098730, -29.5565186, 29.5548477
35: -30.2811527, 1.2470779, -30.2961998, 1.2687111, -30.6081085, 30.5996780
36: -27.1458263, 4.0896454, -27.1364861, 4.0939989, -31.1820602, 31.1702576
37: -38.9764214, -2.0026684, -38.9821167, -1.9962730, -36.4897537, 36.5049820
38: -32.2020607, 3.8129454, -32.2035141, 3.8151274, -36.0171890, 36.0164604
39: -37.8044281, 4.4250069, -37.8047676, 4.4473972, -42.1972198, 42.1753998
40: -30.2019234, 4.3766584, -30.1890717, 4.3752375, -34.5771599, 34.5657310
41: -21.4257317, 5.8672104, -21.3878136, 5.8469443, -27.1983643, 27.1802368
42: -12.4982891, 7.0976524, -12.4578447, 7.0826998, -19.5809898, 19.5554962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1705

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2580953, upper bound: 12.1709473
time: 42.44 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2581817, upper bound: 12.2067042
time: 32.42 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -22.5977745, 10.1334076, -22.6147709, 10.1396561, -32.7374306, 32.7481766
1: -9.0541630, 13.9811420, -9.0639048, 13.9871826, -23.0413456, 23.0450478
2: -8.0799713, 12.9090061, -8.0874844, 12.9117947, -20.7055969, 20.7125092
3: -9.3903713, 14.5744829, -9.4000893, 14.5868835, -23.9772549, 23.9745712
4: -11.1040287, 13.9933100, -11.1263123, 14.0116768, -25.0934830, 25.0989761
5: -9.1659060, 14.6410303, -9.1714420, 14.6496201, -23.5654602, 23.5612411
6: -20.4714546, 7.4207296, -20.4975891, 7.4539962, -27.9254513, 27.9183197
7: -11.2848463, 16.7829685, -11.2924099, 16.7900982, -27.5214920, 27.5213852
8: -13.3558226, 17.4001808, -13.3762360, 17.4173622, -30.6604919, 30.6654129
9: -6.9590163, 16.0778446, -6.9905481, 16.1105404, -23.0695572, 23.0683937
10: -15.3175716, 19.5919418, -15.3265572, 19.6083164, -34.9258881, 34.9184990
11: -17.7677975, 12.7545481, -17.7907124, 12.7749987, -30.5427971, 30.5452614
12: -21.9998875, 9.5517941, -22.0469170, 9.5992146, -30.1894913, 30.1781311
13: -16.9397964, 14.1633978, -16.9531746, 14.1832476, -31.1230431, 31.1165733
14: -35.5945053, 5.6648893, -35.6348572, 5.6996326, -41.2941360, 41.2997475
15: -14.0734558, 10.5142851, -14.0929670, 10.5177383, -24.5911942, 24.6072521
16: -17.5769081, 14.2198391, -17.5972557, 14.2428093, -31.8197174, 31.8170948
17: -38.8900223, 10.3399868, -38.9540825, 10.4014845, -49.2915077, 49.2940674
18: -19.2733116, 7.6648488, -19.2897282, 7.6730847, -26.9463959, 26.9545765
19: -15.6449394, 3.5612657, -15.6443901, 3.5660346, -19.2109737, 19.2056561
20: -11.3733368, 7.3633032, -11.3948555, 7.3822298, -18.7555656, 18.7581596
21: -17.7069550, 6.7359743, -17.7179852, 6.7460575, -24.4530125, 24.4539604
22: -20.7301846, 6.4389248, -20.7627048, 6.4688263, -27.1990108, 27.2016296
23: -14.2638083, 5.9740734, -14.2767944, 5.9831762, -20.2469845, 20.2508678
24: -17.5103855, 7.5315390, -17.5225983, 7.5352449, -25.0456314, 25.0541382
25: -14.8121138, 7.4325280, -14.8292017, 7.4495640, -22.2616768, 22.2617302
26: -21.2913837, 10.0939159, -21.3215313, 10.1104898, -31.4018745, 31.4154472
27: -17.5331459, 8.2797041, -17.5507965, 8.2868309, -25.8199768, 25.8305016
28: -14.3786736, 7.1325550, -14.3866692, 7.1376309, -21.5163040, 21.5192242
29: -21.8446407, 8.5979614, -21.8961525, 8.6454468, -30.4900875, 30.4941139
30: -16.5223808, 9.8076477, -16.5322189, 9.8149738, -26.3373547, 26.3398666
31: -19.2461262, 5.6934171, -19.2546062, 5.7038240, -24.9499512, 24.9480228
32: -19.1454201, 8.2211952, -19.1708527, 8.2478008, -27.3932209, 27.3920479
33: -33.5397415, 4.6149693, -33.5880394, 4.6582832, -37.7315063, 37.7203903
34: -31.4701500, -0.9311733, -31.5070381, -0.9003735, -29.5804062, 29.5885010
35: -30.2919292, 1.2496262, -30.3273163, 1.2782688, -30.6305618, 30.6352615
36: -27.1581917, 4.0919161, -27.1710739, 4.1051311, -31.2191315, 31.2103653
37: -38.9896011, -1.9981041, -39.0202103, -1.9804192, -36.5470734, 36.5547256
38: -32.2107506, 3.8180370, -32.2292252, 3.8337069, -36.0444565, 36.0472641
39: -37.8279686, 4.4278469, -37.8708115, 4.4655018, -42.2399597, 42.2394943
40: -30.2273216, 4.3796883, -30.2616081, 4.4050055, -34.6323280, 34.6412964
41: -21.4511623, 5.8729267, -21.4604607, 5.8813858, -27.2582397, 27.2584457
42: -12.5306683, 7.1038294, -12.5513182, 7.1277428, -19.6584110, 19.6551476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1705

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2599827, upper bound: 12.2030663
time: 48.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2600386, upper bound: 12.2386695
time: 50.18 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.6127853, 10.1344147, -22.6001511, 10.1116333, -32.7244186, 32.7345657
1: -9.0630302, 13.9845848, -9.0544968, 13.9731674, -23.0361977, 23.0390816
2: -8.0857725, 12.9081020, -8.0789757, 12.8954372, -20.6975937, 20.7039185
3: -9.3940477, 14.5894041, -9.3699780, 14.5823736, -23.9764214, 23.9593811
4: -11.1229267, 14.0171394, -11.1129971, 13.9912033, -25.0908813, 25.0998383
5: -9.1683550, 14.6484900, -9.1576023, 14.6366959, -23.5560150, 23.5583038
6: -20.4871540, 7.4533305, -20.4382362, 7.4246039, -27.9117584, 27.8915672
7: -11.2914524, 16.7887630, -11.2800608, 16.7814865, -27.5192490, 27.5150757
8: -13.3743629, 17.4202995, -13.3657656, 17.4045029, -30.6716309, 30.6759567
9: -6.9697051, 16.1247616, -6.9239511, 16.0980320, -23.0677376, 23.0487137
10: -15.3203611, 19.6049442, -15.2973461, 19.5949287, -34.9152908, 34.9022903
11: -17.7927418, 12.7625589, -17.7676468, 12.7323704, -30.5251122, 30.5302048
12: -22.0480728, 9.5973263, -22.0073471, 9.5741978, -30.1762390, 30.1793671
13: -16.9237595, 14.1810532, -16.8593521, 14.1353798, -31.0591393, 31.0404053
14: -35.6447220, 5.6640148, -35.5971489, 5.5914135, -41.2361374, 41.2611618
15: -14.0918541, 10.5138321, -14.0816040, 10.4956188, -24.5874729, 24.5954361
16: -17.5933380, 14.2351589, -17.5695953, 14.2362280, -31.8295670, 31.8047543
17: -38.9769707, 10.3583422, -38.9221153, 10.2692118, -49.2461815, 49.2804565
18: -19.2902641, 7.6293797, -19.2396851, 7.5449667, -26.8352318, 26.8690643
19: -15.6435127, 3.5483909, -15.6195498, 3.5144892, -19.1580009, 19.1679401
20: -11.3992634, 7.3720722, -11.3825512, 7.3527975, -18.7520599, 18.7546234
21: -17.7166176, 6.7284770, -17.6915092, 6.6953750, -24.4119930, 24.4199867
22: -20.7680054, 6.4476414, -20.7378883, 6.4008904, -27.1688957, 27.1855297
23: -14.2776222, 5.9651294, -14.2530422, 5.9292049, -20.2068272, 20.2181721
24: -17.5209675, 7.5040798, -17.4766273, 7.4387531, -24.9597206, 24.9807072
25: -14.8316631, 7.4299655, -14.8048229, 7.3863139, -22.2179775, 22.2347889
26: -21.3258667, 10.0806217, -21.2821884, 10.0204182, -31.3462849, 31.3628101
27: -17.5503120, 8.2570915, -17.5085163, 8.2026224, -25.7529335, 25.7656078
28: -14.3847332, 7.1208525, -14.3624649, 7.0870767, -21.4718094, 21.4833183
29: -21.9145775, 8.6253090, -21.8915710, 8.5817833, -30.4963608, 30.5168800
30: -16.5290699, 9.7962265, -16.5022411, 9.7570877, -26.2861576, 26.2984676
31: -19.2517776, 5.6772056, -19.2149582, 5.6247902, -24.8765678, 24.8921642
32: -19.1531677, 8.2452259, -19.0906830, 8.2084875, -27.3616562, 27.3359089
33: -33.5734711, 4.6758127, -33.5320892, 4.6580219, -37.7610245, 37.7283707
34: -31.5006409, -0.8889484, -31.4794064, -0.8948164, -29.6134567, 29.5926666
35: -30.3213806, 1.2896214, -30.3003941, 1.2834454, -30.6630707, 30.6466751
36: -27.1652641, 4.1045189, -27.1423416, 4.0960445, -31.1991882, 31.1905746
37: -39.0145988, -1.9800968, -38.9886246, -1.9885979, -36.5668335, 36.5391083
38: -32.2254639, 3.8333511, -32.2112656, 3.8204966, -36.0459595, 36.0446167
39: -37.8531647, 4.4811268, -37.8091660, 4.4675331, -42.2660675, 42.2323456
40: -30.2429962, 4.4135427, -30.1942673, 4.3878860, -34.6308823, 34.6078110
41: -21.4396629, 5.8784561, -21.3910618, 5.8497777, -27.2151337, 27.1950531
42: -12.5312977, 7.1246400, -12.4685192, 7.0852175, -19.6165161, 19.5931587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1705

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2585360, upper bound: 12.1921571
time: 36.24 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2581817, upper bound: 12.2285073
time: 37.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.6198196, 10.1454220, -22.6192608, 10.1422882, -32.7621078, 32.7646828
1: -9.0668526, 13.9907589, -9.0667152, 13.9895926, -23.0564461, 23.0574741
2: -8.0897598, 12.9149408, -8.0894861, 12.9133444, -20.7187500, 20.7207031
3: -9.4045954, 14.5928955, -9.4016676, 14.5931110, -23.9977074, 23.9945641
4: -11.1295290, 14.0288086, -11.1288900, 14.0241241, -25.1314468, 25.1321487
5: -9.1732082, 14.6532784, -9.1736536, 14.6533890, -23.5774536, 23.5781784
6: -20.5125542, 7.4576707, -20.5114059, 7.4577913, -27.9703445, 27.9690762
7: -11.2964249, 16.7927399, -11.2956972, 16.7922153, -27.5354080, 27.5353394
8: -13.3786144, 17.4289017, -13.3782578, 17.4262199, -30.6971359, 30.6967697
9: -6.9933815, 16.1275902, -6.9929214, 16.1288414, -23.1222229, 23.1080093
10: -15.3319168, 19.6108418, -15.3301115, 19.6144600, -34.9463768, 34.9409523
11: -17.8013535, 12.7782173, -17.8017902, 12.7764702, -30.5778236, 30.5800076
12: -22.0699615, 9.6043348, -22.0704880, 9.6036949, -30.2507477, 30.2539902
13: -16.9584503, 14.1870270, -16.9584351, 14.1865797, -31.1450310, 31.1454620
14: -35.6541595, 5.7041740, -35.6544495, 5.7014046, -41.3555641, 41.3586235
15: -14.0964947, 10.5227299, -14.0966091, 10.5206146, -24.6171093, 24.6193390
16: -17.6037006, 14.2403011, -17.6037483, 14.2508125, -31.8545132, 31.8440495
17: -38.9854202, 10.4066362, -38.9866180, 10.4039936, -49.3894119, 49.3932533
18: -19.2959538, 7.6749701, -19.2968330, 7.6742401, -26.9701939, 26.9718037
19: -15.6478357, 3.5666246, -15.6476707, 3.5666914, -19.2145271, 19.2142944
20: -11.4044819, 7.3832283, -11.4046526, 7.3834953, -18.7879772, 18.7878799
21: -17.7232819, 6.7477660, -17.7232704, 6.7472281, -24.4705105, 24.4710369
22: -20.7741890, 6.4719028, -20.7767773, 6.4706249, -27.2448139, 27.2486801
23: -14.2824917, 5.9851484, -14.2826347, 5.9845152, -20.2670059, 20.2677841
24: -17.5278854, 7.5379820, -17.5276413, 7.5366688, -25.0645542, 25.0656242
25: -14.8370266, 7.4532046, -14.8376236, 7.4513698, -22.2883968, 22.2908287
26: -21.3341866, 10.1127720, -21.3350353, 10.1113873, -31.4455738, 31.4478073
27: -17.5571804, 8.2878895, -17.5580139, 8.2882671, -25.8454475, 25.8459034
28: -14.3898220, 7.1388617, -14.3897896, 7.1384678, -21.5282898, 21.5286522
29: -21.9203682, 8.6485214, -21.9211273, 8.6468925, -30.5672607, 30.5696487
30: -16.5361080, 9.8179226, -16.5362606, 9.8170786, -26.3531876, 26.3541832
31: -19.2584038, 5.7058988, -19.2579422, 5.7052221, -24.9636269, 24.9638405
32: -19.1862068, 8.2507324, -19.1842232, 8.2507601, -27.4369659, 27.4349556
33: -33.5944901, 4.6794090, -33.5935135, 4.6804104, -37.8083038, 37.7804337
34: -31.5116730, -0.8861361, -31.5110378, -0.8853130, -29.6373062, 29.6262741
35: -30.3321342, 1.2921562, -30.3315201, 1.2930355, -30.6855316, 30.6822128
36: -27.1776409, 4.1067705, -27.1769638, 4.1072245, -31.2362747, 31.2306366
37: -39.0278397, -1.9755325, -39.0266991, -1.9727364, -36.6241989, 36.5888290
38: -32.2341690, 3.8384008, -32.2370453, 3.8390837, -36.0732536, 36.0754471
39: -37.8767738, 4.4839439, -37.8751907, 4.4856453, -42.3088379, 42.2965088
40: -30.2683868, 4.4165573, -30.2668190, 4.4176311, -34.6860199, 34.6833763
41: -21.4651222, 5.8841591, -21.4637108, 5.8842020, -27.2750473, 27.2732773
42: -12.5636711, 7.1307883, -12.5619965, 7.1302595, -19.6939316, 19.6927853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1705

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2604255, upper bound: 12.2242862
time: 41.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2604797, upper bound: 12.2604784
time: 40.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 83.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 83.96
Output dim: 9, lower bound: -12.2580953, upper bound: 12.1709473
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 83.96
Output dim: 9, lower bound: -12.2581817, upper bound: 12.2067042
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 83.96
Output dim: 9, lower bound: -12.2599827, upper bound: 12.2030663
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 83.96
Output dim: 9, lower bound: -12.2600386, upper bound: 12.2386695
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 83.96
Output dim: 9, lower bound: -12.2585360, upper bound: 12.1921571
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 83.96
Output dim: 9, lower bound: -12.2581817, upper bound: 12.2285073
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 83.96
Output dim: 9, lower bound: -12.2604255, upper bound: 12.2242862
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 83.96
Output dim: 9, lower bound: -12.2604797, upper bound: 12.2604784

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -22.5503616, 10.0960083, -22.5919132, 10.1017227, -32.6520844, 32.6879196
1: -9.0299664, 13.9631319, -9.0495014, 13.9678144, -22.9977798, 23.0126343
2: -8.0663643, 12.8970470, -8.0754290, 12.8926773, -20.6727600, 20.6881409
3: -9.3719358, 14.5609264, -9.3670006, 14.5734625, -23.9453983, 23.9279270
4: -11.0712090, 13.9568043, -11.1078720, 13.9711285, -25.0191498, 25.0392380
5: -9.1498747, 14.6271267, -9.1527147, 14.6302633, -23.5282211, 23.5268478
6: -20.4212265, 7.3903899, -20.4175110, 7.4174790, -27.8387051, 27.8079014
7: -11.2673616, 16.7719650, -11.2746897, 16.7778625, -27.4909973, 27.4904556
8: -13.3338890, 17.3757553, -13.3615665, 17.3918648, -30.6130066, 30.6256790
9: -6.8932061, 16.0311928, -6.9193978, 16.0659409, -22.9591465, 22.9505901
10: -15.2770033, 19.5650845, -15.2906504, 19.5834770, -34.8604813, 34.8557358
11: -17.7250328, 12.7093420, -17.7467213, 12.7294664, -30.4544983, 30.4560623
12: -21.9407082, 9.5057735, -21.9727993, 9.5667381, -30.0742111, 30.0536957
13: -16.8877754, 14.1434479, -16.8503761, 14.1296196, -31.0173950, 30.9938240
14: -35.5571747, 5.5970860, -35.5702744, 5.5879984, -41.1451721, 41.1673584
15: -14.0445967, 10.4893465, -14.0746555, 10.4882250, -24.5328217, 24.5640030
16: -17.5219803, 14.1752300, -17.5588779, 14.2169075, -31.7388878, 31.7341080
17: -38.8089104, 10.2126179, -38.8679810, 10.2641811, -49.0730896, 49.0805969
18: -19.2580891, 7.6048226, -19.2297401, 7.5408955, -26.7989845, 26.8345623
19: -15.6234436, 3.5317082, -15.6123676, 3.5128648, -19.1363087, 19.1440754
20: -11.3375216, 7.3288574, -11.3643036, 7.3502555, -18.6877766, 18.6931610
21: -17.6661854, 6.6897478, -17.6774521, 6.6930838, -24.3592682, 24.3671989
22: -20.6761799, 6.3718147, -20.7106781, 6.3976994, -27.0738792, 27.0824928
23: -14.2424259, 5.9451723, -14.2430563, 5.9267511, -20.1691780, 20.1882286
24: -17.4787807, 7.4851274, -17.4653435, 7.4362707, -24.9150505, 24.9504700
25: -14.7703457, 7.3820915, -14.7858219, 7.3831129, -22.1534576, 22.1679134
26: -21.2574692, 10.0434771, -21.2622585, 10.0182676, -31.2757378, 31.3057365
27: -17.5137062, 8.2366056, -17.4981747, 8.1995068, -25.7132130, 25.7347794
28: -14.3608952, 7.1056833, -14.3563519, 7.0850668, -21.4459610, 21.4620361
29: -21.7790031, 8.5145111, -21.8495140, 8.5788727, -30.3578758, 30.3640251
30: -16.4771290, 9.7522411, -16.4876556, 9.7530689, -26.2301979, 26.2398968
31: -19.2109489, 5.6460228, -19.2047005, 5.6221733, -24.8331223, 24.8507233
32: -19.0973930, 8.2026176, -19.0738163, 8.2032137, -27.3006058, 27.2764339
33: -33.4881210, 4.5965805, -33.5228462, 4.6314583, -37.6482162, 37.6497650
34: -31.4117279, -0.9791203, -31.4731674, -0.9231319, -29.4930649, 29.5052032
35: -30.2451782, 1.2147942, -30.2937469, 1.2586784, -30.5594711, 30.5645142
36: -27.1333313, 4.0720654, -27.1345024, 4.0901871, -31.1558838, 31.1472778
37: -38.9384613, -2.0240631, -38.9769936, -2.0027237, -36.4380493, 36.4755707
38: -32.1712837, 3.7784433, -32.2007828, 3.8045683, -35.9758530, 35.9792252
39: -37.7526703, 4.3902187, -37.8007088, 4.4360447, -42.1336365, 42.1365204
40: -30.1664982, 4.3510485, -30.1852245, 4.3676662, -34.5341644, 34.5362740
41: -21.4027348, 5.8426571, -21.3855038, 5.8404059, -27.1681595, 27.1530533
42: -12.4702911, 7.0747600, -12.4502993, 7.0807304, -19.5510216, 19.5250587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 633

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2256776, upper bound: 12.1685182
time: 36.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2256776, upper bound: 12.1685182
time: 47.98 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -22.5877895, 10.1193123, -22.5946484, 10.1079426, -32.6957321, 32.7139587
1: -9.0492077, 13.9728546, -9.0512981, 13.9699917, -23.0191994, 23.0241528
2: -8.0746479, 12.9002190, -8.0765152, 12.8932133, -20.6819077, 20.6925354
3: -9.3771782, 14.5688505, -9.3675003, 14.5753889, -23.9525681, 23.9363518
4: -11.0953321, 13.9797134, -11.1097031, 13.9781075, -25.0502014, 25.0630951
5: -9.1560135, 14.6348991, -9.1537113, 14.6324406, -23.5381699, 23.5383224
6: -20.4395790, 7.4136229, -20.4222908, 7.4198432, -27.8594227, 27.8359146
7: -11.2784519, 16.7774372, -11.2762766, 16.7788239, -27.5030365, 27.4983826
8: -13.3498936, 17.3893528, -13.3631401, 17.3948784, -30.6324844, 30.6411438
9: -6.9317961, 16.0728111, -6.9203396, 16.0789909, -23.0107880, 22.9931507
10: -15.3032436, 19.5823898, -15.2927771, 19.5875683, -34.8908119, 34.8751678
11: -17.7562008, 12.7357531, -17.7556057, 12.7298594, -30.4860611, 30.4913597
12: -21.9757061, 9.5421886, -21.9829845, 9.5688238, -30.1084290, 30.0999756
13: -16.9019432, 14.1545553, -16.8530331, 14.1310806, -31.0330238, 31.0075874
14: -35.5820236, 5.6230078, -35.5765305, 5.5890474, -41.1710701, 41.1995392
15: -14.0647926, 10.5019445, -14.0765686, 10.4916019, -24.5563946, 24.5785141
16: -17.5630798, 14.2079210, -17.5618744, 14.2259541, -31.7890339, 31.7697945
17: -38.8775024, 10.2875252, -38.8881989, 10.2652264, -49.1427307, 49.1757240
18: -19.2654839, 7.6116109, -19.2318687, 7.5411968, -26.8066807, 26.8434792
19: -15.6382866, 3.5406446, -15.6154804, 3.5129981, -19.1512852, 19.1561241
20: -11.3658295, 7.3497782, -11.3719759, 7.3507247, -18.7165546, 18.7217541
21: -17.6972580, 6.7143545, -17.6851749, 6.6934147, -24.3906727, 24.3995285
22: -20.7209625, 6.4132061, -20.7227821, 6.3985929, -27.1195564, 27.1359882
23: -14.2571144, 5.9522610, -14.2465696, 5.9272499, -20.1843643, 20.1988297
24: -17.5001945, 7.4959936, -17.4704342, 7.4367666, -24.9369621, 24.9664268
25: -14.8033972, 7.4070601, -14.7952080, 7.3837218, -22.1871185, 22.2022686
26: -21.2803345, 10.0601168, -21.2677841, 10.0189953, -31.2993298, 31.3278999
27: -17.5237045, 8.2448101, -17.5004082, 8.1998138, -25.7235184, 25.7452183
28: -14.3715000, 7.1117907, -14.3586521, 7.0853157, -21.4568157, 21.4704437
29: -21.8352699, 8.5726261, -21.8654442, 8.5795937, -30.4148636, 30.4380703
30: -16.5124092, 9.7833834, -16.4971657, 9.7540855, -26.2664948, 26.2805481
31: -19.2370892, 5.6629210, -19.2108231, 5.6227808, -24.8598709, 24.8737450
32: -19.1087093, 8.2131643, -19.0760708, 8.2046871, -27.3133965, 27.2892342
33: -33.5164833, 4.6083193, -33.5258522, 4.6348381, -37.6811752, 37.6641617
34: -31.4560223, -0.9366474, -31.4743900, -0.9107780, -29.5529938, 29.5487747
35: -30.2788754, 1.2447376, -30.2954102, 1.2678719, -30.6066360, 30.5957413
36: -27.1439762, 4.0773239, -27.1358147, 4.0896378, -31.1767883, 31.1585159
37: -38.9741364, -2.0097713, -38.9813347, -1.9986572, -36.4890137, 36.4991150
38: -32.1995163, 3.8046131, -32.2026596, 3.8121653, -36.0116806, 36.0072708
39: -37.8019142, 4.4221611, -37.8039169, 4.4464760, -42.1939697, 42.1715393
40: -30.1999092, 4.3731070, -30.1883774, 4.3740883, -34.5739975, 34.5614853
41: -21.4234543, 5.8582072, -21.3870411, 5.8438773, -27.1930466, 27.1701508
42: -12.4947138, 7.0953274, -12.4566011, 7.0819092, -19.5766220, 19.5519295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 633

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2257921, upper bound: 12.2047219
time: 53.76 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2560566, upper bound: 12.2047219
time: 41.84 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.5573902, 10.1070004, -22.6110153, 10.1323376, -32.6897278, 32.7180176
1: -9.0337734, 13.9692783, -9.0617065, 13.9842348, -23.0180092, 23.0309849
2: -8.0703344, 12.9038620, -8.0859137, 12.9105864, -20.6939392, 20.7048988
3: -9.3825169, 14.5644245, -9.3986816, 14.5842037, -23.9667206, 23.9631062
4: -11.0778303, 13.9684219, -11.1237850, 14.0040646, -25.0596924, 25.0715561
5: -9.1546993, 14.6318951, -9.1687946, 14.6469803, -23.5496216, 23.5466919
6: -20.4466419, 7.3947392, -20.4906960, 7.4506388, -27.8972816, 27.8854351
7: -11.2723713, 16.7759228, -11.2903252, 16.7886200, -27.5071487, 27.5106888
8: -13.3381357, 17.3843460, -13.3740892, 17.4136028, -30.6384506, 30.6464767
9: -6.9168639, 16.0340424, -6.9883833, 16.0967598, -23.0136242, 23.0223885
10: -15.2885265, 19.5709743, -15.3234320, 19.6030006, -34.8915253, 34.8944054
11: -17.7336102, 12.7250204, -17.7808666, 12.7735767, -30.5071869, 30.5058861
12: -21.9626122, 9.5128288, -22.0359230, 9.5963144, -30.1487808, 30.1283340
13: -16.9224930, 14.1494007, -16.9494209, 14.1808624, -31.1033554, 31.0988216
14: -35.5666428, 5.6372023, -35.6275215, 5.6979799, -41.2646217, 41.2647247
15: -14.0492306, 10.4982462, -14.0896740, 10.5132236, -24.5624542, 24.5879211
16: -17.5323486, 14.1803799, -17.5930424, 14.2314358, -31.7637844, 31.7734222
17: -38.8173218, 10.2609615, -38.9325180, 10.3989773, -49.2163010, 49.1934814
18: -19.2637672, 7.6503844, -19.2868729, 7.6701479, -26.9339142, 26.9372578
19: -15.6277657, 3.5499468, -15.6404867, 3.5650997, -19.1928654, 19.1904335
20: -11.3427258, 7.3400302, -11.3863983, 7.3809738, -18.7236996, 18.7264290
21: -17.6728592, 6.7090263, -17.7092381, 6.7449379, -24.4177971, 24.4182644
22: -20.6823349, 6.3961239, -20.7495575, 6.4674473, -27.1497822, 27.1456814
23: -14.2472534, 5.9651842, -14.2726393, 5.9820905, -20.2293434, 20.2378235
24: -17.4856453, 7.5190420, -17.5163574, 7.5341740, -25.0198193, 25.0354004
25: -14.7756824, 7.4052806, -14.8186398, 7.4481769, -22.2238598, 22.2239208
26: -21.2657623, 10.0757084, -21.3150558, 10.1092243, -31.3749866, 31.3907642
27: -17.5205326, 8.2673931, -17.5476799, 8.2851620, -25.8056946, 25.8150730
28: -14.3659573, 7.1236901, -14.3836679, 7.1364660, -21.5024223, 21.5073586
29: -21.7847862, 8.5376701, -21.8790340, 8.6440582, -30.4288445, 30.4167042
30: -16.4840755, 9.7738934, -16.5216999, 9.8130941, -26.2971687, 26.2955933
31: -19.2175903, 5.6747217, -19.2477074, 5.7026062, -24.9201965, 24.9224281
32: -19.1304588, 8.2081413, -19.1673813, 8.2455215, -27.3759804, 27.3755226
33: -33.5091400, 4.6001911, -33.5842018, 4.6539001, -37.6954956, 37.7018738
34: -31.4227295, -0.9763374, -31.5047913, -0.9136419, -29.5169525, 29.5388336
35: -30.2559414, 1.2173100, -30.3248711, 1.2682610, -30.5819397, 30.6000824
36: -27.1456509, 4.0743713, -27.1690998, 4.1013184, -31.1929626, 31.1873932
37: -38.9515953, -2.0195141, -39.0150986, -1.9868736, -36.4952698, 36.5252533
38: -32.1800346, 3.7834711, -32.2265396, 3.8231220, -36.0031586, 36.0100098
39: -37.7762184, 4.3930187, -37.8667145, 4.4540997, -42.1764069, 42.2005463
40: -30.1918564, 4.3541260, -30.2577209, 4.3974247, -34.5892792, 34.6118469
41: -21.4281731, 5.8483467, -21.4581490, 5.8748398, -27.2280731, 27.2312622
42: -12.5026646, 7.0809598, -12.5437813, 7.1258135, -19.6284790, 19.6247406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 633

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2256776, upper bound: 12.2007122
time: 47.84 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2579348, upper bound: 12.2007122
time: 35.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.5948315, 10.1302776, -22.6137657, 10.1385708, -32.7334023, 32.7440414
1: -9.0530109, 13.9790363, -9.0635071, 13.9864197, -23.0394306, 23.0425434
2: -8.0786457, 12.9070435, -8.0870104, 12.9111366, -20.7030869, 20.7092972
3: -9.3877392, 14.5723286, -9.3991861, 14.5861626, -23.9739017, 23.9715157
4: -11.1019573, 13.9913292, -11.1255836, 14.0110178, -25.0907440, 25.0953903
5: -9.1608543, 14.6396666, -9.1697826, 14.6491337, -23.5595398, 23.5582199
6: -20.4649925, 7.4179721, -20.4954567, 7.4530902, -27.9180832, 27.9134293
7: -11.2834244, 16.7814064, -11.2919083, 16.7895794, -27.5191879, 27.5186386
8: -13.3541260, 17.3979340, -13.3756485, 17.4166069, -30.6579895, 30.6619797
9: -6.9554749, 16.0756340, -6.9893456, 16.1097984, -23.0652733, 23.0633698
10: -15.3147860, 19.5882854, -15.3255758, 19.6071110, -34.9218979, 34.9138603
11: -17.7648258, 12.7513905, -17.7897186, 12.7739611, -30.5387878, 30.5411091
12: -21.9975739, 9.5491943, -22.0461235, 9.5983562, -30.1829834, 30.1745834
13: -16.9366684, 14.1605301, -16.9521599, 14.1822529, -31.1189213, 31.1126900
14: -35.5914536, 5.6631165, -35.6337662, 5.6990261, -41.2904816, 41.2968826
15: -14.0694027, 10.5108461, -14.0915871, 10.5166054, -24.5860081, 24.6024323
16: -17.5734100, 14.2130518, -17.5960503, 14.2405510, -31.8139610, 31.8091011
17: -38.8859482, 10.3358059, -38.9526978, 10.4001036, -49.2860527, 49.2885056
18: -19.2711658, 7.6571813, -19.2890072, 7.6704578, -26.9416237, 26.9461880
19: -15.6426363, 3.5588913, -15.6435966, 3.5652289, -19.2078648, 19.2024879
20: -11.3710403, 7.3609304, -11.3940887, 7.3814344, -18.7524757, 18.7550201
21: -17.7039356, 6.7335877, -17.7169571, 6.7452683, -24.4492035, 24.4505444
22: -20.7271080, 6.4374952, -20.7616348, 6.4683437, -27.1954517, 27.1991310
23: -14.2619572, 5.9722452, -14.2761765, 5.9825625, -20.2445202, 20.2484207
24: -17.5070877, 7.5298958, -17.5214500, 7.5346699, -25.0417576, 25.0513458
25: -14.8087425, 7.4302778, -14.8280182, 7.4487791, -22.2575226, 22.2582970
26: -21.2886543, 10.0923491, -21.3206024, 10.1099253, -31.3985786, 31.4129524
27: -17.5305557, 8.2756090, -17.5499115, 8.2854548, -25.8160095, 25.8255196
28: -14.3765392, 7.1298170, -14.3859587, 7.1366749, -21.5132141, 21.5157757
29: -21.8410664, 8.5958443, -21.8949738, 8.6447392, -30.4858055, 30.4908180
30: -16.5193825, 9.8051014, -16.5312157, 9.8141289, -26.3335114, 26.3363171
31: -19.2437191, 5.6916380, -19.2537918, 5.7032065, -24.9469261, 24.9454308
32: -19.1417637, 8.2186947, -19.1696301, 8.2469482, -27.3887119, 27.3883247
33: -33.5375443, 4.6119032, -33.5872650, 4.6572618, -37.7284622, 37.7162476
34: -31.4670372, -0.9338551, -31.5059586, -0.9012938, -29.5768814, 29.5824432
35: -30.2896557, 1.2472935, -30.3265190, 1.2774220, -30.6290970, 30.6312637
36: -27.1563110, 4.0796251, -27.1704235, 4.1008091, -31.2138824, 31.1985779
37: -38.9873085, -2.0052452, -39.0194664, -1.9827976, -36.5462952, 36.5488815
38: -32.2082367, 3.8096752, -32.2283707, 3.8307133, -36.0389481, 36.0380478
39: -37.8255005, 4.4249725, -37.8699417, 4.4645615, -42.2367401, 42.2356262
40: -30.2252750, 4.3761234, -30.2609138, 4.4037952, -34.6290703, 34.6370392
41: -21.4489040, 5.8638959, -21.4596863, 5.8783340, -27.2529831, 27.2483978
42: -12.5271025, 7.1015048, -12.5500879, 7.1269608, -19.6540642, 19.6515923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 633

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2273797, upper bound: 12.2369001
time: 38.52 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2580511, upper bound: 12.2369001
time: 36.94 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -22.5724411, 10.1080093, -22.5964012, 10.1043320, -32.6767731, 32.7044106
1: -9.0426531, 13.9727116, -9.0523119, 13.9702253, -23.0128784, 23.0250244
2: -8.0761576, 12.9029608, -8.0774212, 12.8942156, -20.6859131, 20.6963196
3: -9.3861217, 14.5793419, -9.3685818, 14.5796947, -23.9658165, 23.9479237
4: -11.0967388, 13.9923153, -11.1104746, 13.9835720, -25.0570984, 25.0724411
5: -9.1571770, 14.6393833, -9.1549215, 14.6340609, -23.5401993, 23.5437698
6: -20.4623337, 7.4273844, -20.4313450, 7.4212446, -27.8835793, 27.8587303
7: -11.2789602, 16.7817478, -11.2779808, 16.7800179, -27.5049210, 27.5044250
8: -13.3567142, 17.4044571, -13.3636379, 17.4006824, -30.6496582, 30.6570511
9: -6.9276109, 16.0809746, -6.9217787, 16.0842876, -23.0118980, 23.0027542
10: -15.2913666, 19.5839195, -15.2942467, 19.5896397, -34.8810043, 34.8781662
11: -17.7585182, 12.7330561, -17.7577972, 12.7309017, -30.4894199, 30.4908524
12: -22.0107937, 9.5584011, -21.9963303, 9.5712709, -30.1355438, 30.1295166
13: -16.9065113, 14.1670866, -16.8555946, 14.1329908, -31.0395012, 31.0226822
14: -35.6168175, 5.6363020, -35.5899048, 5.5897894, -41.2066078, 41.2262077
15: -14.0676832, 10.4977446, -14.0783205, 10.4910965, -24.5587807, 24.5760651
16: -17.5487213, 14.1956806, -17.5653877, 14.2249002, -31.7736206, 31.7610683
17: -38.9042969, 10.2792387, -38.9005432, 10.2667313, -49.1710281, 49.1797829
18: -19.2806854, 7.6149344, -19.2368221, 7.5420318, -26.8227177, 26.8517570
19: -15.6263466, 3.5370743, -15.6156483, 3.5135496, -19.1398964, 19.1527233
20: -11.3686438, 7.3487821, -11.3741026, 7.3515239, -18.7201672, 18.7228851
21: -17.6824722, 6.7015657, -17.6827507, 6.6942325, -24.3767052, 24.3843155
22: -20.7201233, 6.4048176, -20.7247314, 6.3995104, -27.1196327, 27.1295490
23: -14.2610474, 5.9562654, -14.2488918, 5.9280949, -20.1891422, 20.2051582
24: -17.4961700, 7.4915867, -17.4703846, 7.4376783, -24.9338493, 24.9619713
25: -14.7952328, 7.4027591, -14.7942333, 7.3849330, -22.1801662, 22.1969929
26: -21.3002110, 10.0623016, -21.2757702, 10.0191936, -31.3194046, 31.3380718
27: -17.5376892, 8.2447395, -17.5053864, 8.2009439, -25.7386322, 25.7501259
28: -14.3720198, 7.1120052, -14.3594589, 7.0859041, -21.4579239, 21.4714642
29: -21.8547516, 8.5650578, -21.8744659, 8.5803566, -30.4351082, 30.4395237
30: -16.4907398, 9.7625446, -16.4917030, 9.7551861, -26.2459259, 26.2542477
31: -19.2232552, 5.6585202, -19.2080326, 5.6235733, -24.8468285, 24.8665524
32: -19.1381836, 8.2322025, -19.0872059, 8.2061834, -27.3443680, 27.3194084
33: -33.5427933, 4.6610708, -33.5283127, 4.6535845, -37.7249832, 37.7097549
34: -31.4532795, -0.9341335, -31.4771500, -0.9080858, -29.5499268, 29.5429993
35: -30.2853394, 1.2573414, -30.2979507, 1.2734432, -30.6143875, 30.6115494
36: -27.1527691, 4.0868855, -27.1403599, 4.0922623, -31.1730881, 31.1674805
37: -38.9767380, -2.0015182, -38.9835281, -1.9950981, -36.5151215, 36.5095367
38: -32.1947861, 3.7987375, -32.2085953, 3.8098860, -36.0046730, 36.0073318
39: -37.8014412, 4.4463539, -37.8050995, 4.4562149, -42.2024994, 42.1934357
40: -30.2075691, 4.3879280, -30.1904831, 4.3802834, -34.5878525, 34.5784111
41: -21.4166698, 5.8538723, -21.3887634, 5.8432446, -27.1849289, 27.1678772
42: -12.5032787, 7.1017280, -12.4609699, 7.0832605, -19.5865402, 19.5626984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 666

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2367241, upper bound: 12.1919055
time: 60.67 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2367241, upper bound: 12.1921576
time: 39.42 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.6098442, 10.1312523, -22.5991364, 10.1105442, -32.7203903, 32.7303886
1: -9.0618801, 13.9824467, -9.0541019, 13.9724159, -23.0342960, 23.0365486
2: -8.0844383, 12.9061604, -8.0785131, 12.8947926, -20.6950531, 20.7006950
3: -9.3913612, 14.5872707, -9.3690834, 14.5816584, -23.9730186, 23.9563541
4: -11.1208534, 14.0152493, -11.1122732, 13.9905605, -25.0881042, 25.0962601
5: -9.1633263, 14.6471462, -9.1559238, 14.6362152, -23.5501633, 23.5552673
6: -20.4806747, 7.4505873, -20.4360962, 7.4236531, -27.9043274, 27.8866844
7: -11.2900047, 16.7872257, -11.2795887, 16.7809410, -27.5169678, 27.5123901
8: -13.3726425, 17.4180756, -13.3651943, 17.4037323, -30.6690903, 30.6725082
9: -6.9661875, 16.1225777, -6.9227099, 16.0973148, -23.0635033, 23.0452881
10: -15.3175306, 19.6012611, -15.2963543, 19.5937538, -34.9112854, 34.8976135
11: -17.7897453, 12.7593985, -17.7666492, 12.7313023, -30.5210476, 30.5260468
12: -22.0458031, 9.5947514, -22.0065918, 9.5733414, -30.1697693, 30.1757889
13: -16.9206295, 14.1781883, -16.8582935, 14.1343880, -31.0550175, 31.0364819
14: -35.6416893, 5.6622429, -35.5961113, 5.5908566, -41.2325439, 41.2583542
15: -14.0878487, 10.5103540, -14.0802174, 10.4944582, -24.5823059, 24.5905724
16: -17.5898361, 14.2284031, -17.5683899, 14.2340012, -31.8238373, 31.7967930
17: -38.9728584, 10.3541317, -38.9207077, 10.2677736, -49.2406311, 49.2748413
18: -19.2881279, 7.6217222, -19.2389698, 7.5423498, -26.8304787, 26.8606911
19: -15.6412125, 3.5460110, -15.6187649, 3.5136528, -19.1548653, 19.1647758
20: -11.3969727, 7.3696871, -11.3817844, 7.3519955, -18.7489681, 18.7514725
21: -17.7135658, 6.7261133, -17.6904850, 6.6945562, -24.4081230, 24.4165993
22: -20.7649078, 6.4461899, -20.7368431, 6.4004097, -27.1653175, 27.1830330
23: -14.2757998, 5.9633169, -14.2524376, 5.9285774, -20.2043762, 20.2157555
24: -17.5176659, 7.5023947, -17.4754715, 7.4381857, -24.9558525, 24.9778671
25: -14.8283081, 7.4277287, -14.8036270, 7.3855505, -22.2138596, 22.2313557
26: -21.3231316, 10.0789986, -21.2812653, 10.0198879, -31.3430195, 31.3602638
27: -17.5477180, 8.2529583, -17.5076332, 8.2012253, -25.7489433, 25.7605915
28: -14.3826180, 7.1181164, -14.3617516, 7.0861497, -21.4687672, 21.4798679
29: -21.9110203, 8.6231728, -21.8903923, 8.5810623, -30.4920826, 30.5135651
30: -16.5260849, 9.7936544, -16.5012245, 9.7562189, -26.2823029, 26.2948799
31: -19.2493896, 5.6754293, -19.2141685, 5.6241879, -24.8735771, 24.8895988
32: -19.1494923, 8.2427130, -19.0894241, 8.2076435, -27.3571358, 27.3321381
33: -33.5712357, 4.6728001, -33.5313339, 4.6569653, -37.7579269, 37.7241821
34: -31.4975052, -0.8916254, -31.4783745, -0.8957376, -29.6098709, 29.5866089
35: -30.3191166, 1.2872849, -30.2996330, 1.2826138, -30.6615829, 30.6427078
36: -27.1634197, 4.0922079, -27.1416874, 4.0917568, -31.1940002, 31.1787872
37: -39.0123825, -1.9872360, -38.9878349, -1.9909630, -36.5661163, 36.5332794
38: -32.2229652, 3.8250132, -32.2104378, 3.8175030, -36.0404663, 36.0354500
39: -37.8507576, 4.4783211, -37.8083420, 4.4666443, -42.2628632, 42.2284851
40: -30.2409992, 4.4099579, -30.1935997, 4.3866806, -34.6276779, 34.6035576
41: -21.4373989, 5.8694391, -21.3902874, 5.8466964, -27.2098312, 27.1850128
42: -12.5277519, 7.1222844, -12.4672909, 7.0844221, -19.6121750, 19.5895748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 666

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2368109, upper bound: 12.2280680
time: 58.00 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2368109, upper bound: 12.2285077
time: 32.63 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.5794697, 10.1189709, -22.6155357, 10.1349678, -32.7144394, 32.7345047
1: -9.0464764, 13.9788990, -9.0645218, 13.9866638, -23.0331402, 23.0434208
2: -8.0801401, 12.9097691, -8.0879364, 12.9121494, -20.7070999, 20.7130699
3: -9.3966980, 14.5828209, -9.4002666, 14.5904598, -23.9871578, 23.9830875
4: -11.1033554, 14.0039501, -11.1263781, 14.0165243, -25.0976715, 25.1048126
5: -9.1620140, 14.6441555, -9.1710196, 14.6507902, -23.5616074, 23.5636444
6: -20.4877415, 7.4317241, -20.5045204, 7.4544296, -27.9421711, 27.9362450
7: -11.2839737, 16.7856960, -11.2936516, 16.7907352, -27.5210724, 27.5246582
8: -13.3609257, 17.4130726, -13.3761673, 17.4223938, -30.6751328, 30.6778793
9: -6.9512720, 16.0838013, -6.9907365, 16.1150894, -23.0663605, 23.0613098
10: -15.3029003, 19.5898304, -15.3270302, 19.6091881, -34.9120865, 34.9168625
11: -17.7671642, 12.7486897, -17.7919197, 12.7750359, -30.5422001, 30.5406094
12: -22.0326920, 9.5654297, -22.0595131, 9.6008244, -30.2100754, 30.2041397
13: -16.9412136, 14.1730194, -16.9546833, 14.1842079, -31.1254215, 31.1277027
14: -35.6262283, 5.6764221, -35.6471405, 5.6998034, -41.3260307, 41.3235626
15: -14.0722895, 10.5066624, -14.0933428, 10.5160961, -24.5883865, 24.6000061
16: -17.5591164, 14.2007904, -17.5995369, 14.2394562, -31.7985725, 31.8003273
17: -38.9127502, 10.3275394, -38.9650536, 10.4015064, -49.3142548, 49.2925949
18: -19.2864017, 7.6605010, -19.2940006, 7.6712947, -26.9576969, 26.9545021
19: -15.6306782, 3.5553064, -15.6437550, 3.5657663, -19.1964455, 19.1990623
20: -11.3738470, 7.3599496, -11.3962002, 7.3822422, -18.7560883, 18.7561493
21: -17.6891403, 6.7208118, -17.7145309, 6.7460957, -24.4352360, 24.4353428
22: -20.7262764, 6.4291115, -20.7636299, 6.4692478, -27.1955242, 27.1927414
23: -14.2659130, 5.9762669, -14.2784891, 5.9833918, -20.2493057, 20.2547569
24: -17.5030556, 7.5254688, -17.5214081, 7.5355511, -25.0386066, 25.0468769
25: -14.8005915, 7.4259572, -14.8270798, 7.4500074, -22.2505989, 22.2530365
26: -21.3084984, 10.0944901, -21.3285656, 10.1101103, -31.4186096, 31.4230556
27: -17.5445461, 8.2755079, -17.5549145, 8.2865849, -25.8311310, 25.8304214
28: -14.3770571, 7.1300364, -14.3867512, 7.1372938, -21.5143509, 21.5167885
29: -21.8605461, 8.5882883, -21.9040070, 8.6455002, -30.5060463, 30.4922943
30: -16.4977646, 9.7842140, -16.5257187, 9.8152122, -26.3129768, 26.3099327
31: -19.2298813, 5.6872091, -19.2510204, 5.7040138, -24.9338951, 24.9382286
32: -19.1712475, 8.2376919, -19.1807346, 8.2484779, -27.4197254, 27.4184265
33: -33.5638504, 4.6646709, -33.5897141, 4.6760178, -37.7722321, 37.7618637
34: -31.4642448, -0.9313335, -31.5087719, -0.8985977, -29.5738220, 29.5766907
35: -30.2961273, 1.2598515, -30.3290997, 1.2830105, -30.6368561, 30.6470947
36: -27.1651001, 4.0891418, -27.1749840, 4.1033859, -31.2100906, 31.2075653
37: -38.9898834, -1.9969826, -39.0216179, -1.9792080, -36.5724716, 36.5592651
38: -32.2034912, 3.8038111, -32.2343483, 3.8284731, -36.0319633, 36.0381584
39: -37.8250465, 4.4491119, -37.8711319, 4.4742546, -42.2452240, 42.2575073
40: -30.2329750, 4.3910007, -30.2629890, 4.4100542, -34.6430283, 34.6539917
41: -21.4421005, 5.8595762, -21.4613838, 5.8776541, -27.2448425, 27.2460709
42: -12.5356464, 7.1078987, -12.5544682, 7.1283007, -19.6639481, 19.6623669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 666

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2385993, upper bound: 12.2240329
time: 41.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2367241, upper bound: 12.1921576
time: 47.31 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.6168480, 10.1422653, -22.6182652, 10.1411781, -32.7580261, 32.7605286
1: -9.0656719, 13.9886379, -9.0663023, 13.9888535, -23.0545254, 23.0549393
2: -8.0884199, 12.9129887, -8.0890141, 12.9126968, -20.7162018, 20.7174721
3: -9.4019394, 14.5907593, -9.4007664, 14.5923910, -23.9943314, 23.9915257
4: -11.1274414, 14.0268745, -11.1281891, 14.0234852, -25.1287079, 25.1286316
5: -9.1681566, 14.6519260, -9.1720152, 14.6529350, -23.5715256, 23.5751343
6: -20.5060787, 7.4549298, -20.5092583, 7.4568367, -27.9629154, 27.9641876
7: -11.2950182, 16.7911930, -11.2952328, 16.7916985, -27.5331573, 27.5326233
8: -13.3769112, 17.4266510, -13.3776712, 17.4254818, -30.6946030, 30.6933517
9: -6.9898386, 16.1254005, -6.9916916, 16.1281166, -23.1179543, 23.1023178
10: -15.3290606, 19.6071930, -15.3291826, 19.6132545, -34.9423141, 34.9363747
11: -17.7983780, 12.7750454, -17.8007832, 12.7754030, -30.5737801, 30.5758286
12: -22.0676613, 9.6017485, -22.0697308, 9.6028366, -30.2442856, 30.2503510
13: -16.9553604, 14.1841183, -16.9573803, 14.1856003, -31.1409607, 31.1414986
14: -35.6511040, 5.7023573, -35.6534424, 5.7008276, -41.3519325, 41.3558006
15: -14.0924530, 10.5192823, -14.0952644, 10.5194702, -24.6119232, 24.6145477
16: -17.6001911, 14.2335196, -17.6025448, 14.2485523, -31.8487434, 31.8360634
17: -38.9813080, 10.4024773, -38.9852295, 10.4026451, -49.3839531, 49.3877068
18: -19.2938080, 7.6673121, -19.2961121, 7.6716013, -26.9654083, 26.9634247
19: -15.6455231, 3.5642405, -15.6468906, 3.5659029, -19.2114258, 19.2111320
20: -11.4022007, 7.3808479, -11.4038916, 7.3826904, -18.7848911, 18.7847404
21: -17.7202682, 6.7454014, -17.7222614, 6.7464337, -24.4667015, 24.4676628
22: -20.7710724, 6.4705048, -20.7757378, 6.4701662, -27.2412376, 27.2462425
23: -14.2806578, 5.9833364, -14.2820129, 5.9838820, -20.2645397, 20.2653503
24: -17.5246029, 7.5363007, -17.5265045, 7.5360813, -25.0606842, 25.0628052
25: -14.8336563, 7.4509478, -14.8364382, 7.4506021, -22.2842579, 22.2873859
26: -21.3314209, 10.1111908, -21.3341160, 10.1108360, -31.4422569, 31.4453068
27: -17.5545998, 8.2837496, -17.5571308, 8.2868795, -25.8414803, 25.8408813
28: -14.3876915, 7.1361055, -14.3890800, 7.1375289, -21.5252209, 21.5251846
29: -21.9167976, 8.6463680, -21.9199142, 8.6461678, -30.5629654, 30.5662823
30: -16.5331097, 9.8153820, -16.5352573, 9.8162270, -26.3493366, 26.3506393
31: -19.2560234, 5.7041225, -19.2571449, 5.7046165, -24.9606400, 24.9612675
32: -19.1825447, 8.2482214, -19.1829948, 8.2499199, -27.4324646, 27.4312172
33: -33.5922241, 4.6764116, -33.5927429, 4.6793995, -37.8052368, 37.7762756
34: -31.5085526, -0.8888454, -31.5099850, -0.8862267, -29.6337357, 29.6202393
35: -30.3298531, 1.2897987, -30.3307476, 1.2922134, -30.6840439, 30.6783066
36: -27.1757984, 4.0944576, -27.1763229, 4.1028824, -31.2310333, 31.2188644
37: -39.0255470, -1.9826794, -39.0259705, -1.9751205, -36.6233978, 36.5830002
38: -32.2316933, 3.8300533, -32.2361832, 3.8360949, -36.0677872, 36.0662384
39: -37.8743324, 4.4810963, -37.8743362, 4.4847164, -42.3055725, 42.2926025
40: -30.2663479, 4.4130049, -30.2661018, 4.4164162, -34.6827621, 34.6791077
41: -21.4628372, 5.8751149, -21.4629250, 5.8811378, -27.2697220, 27.2631912
42: -12.5601244, 7.1284342, -12.5607681, 7.1294765, -19.6896019, 19.6892014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 666

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2386704, upper bound: 12.2600370
time: 50.02 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2386704, upper bound: 12.2604788
time: 60.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 112.82 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2256776, upper bound: 12.1685182
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2256776, upper bound: 12.1685182
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2257921, upper bound: 12.2047219
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2560566, upper bound: 12.2047219
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2256776, upper bound: 12.2007122
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2579348, upper bound: 12.2007122
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2273797, upper bound: 12.2369001
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2580511, upper bound: 12.2369001
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2367241, upper bound: 12.1919055
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2367241, upper bound: 12.1921576
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2368109, upper bound: 12.2280680
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2368109, upper bound: 12.2285077
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2385993, upper bound: 12.2240329
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2367241, upper bound: 12.1921576
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2386704, upper bound: 12.2600370
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 112.82
Output dim: 9, lower bound: -12.2386704, upper bound: 12.2604788

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.5862770, 10.1172981, -22.5900536, 10.1016846, -32.6879616, 32.7073517
1: -9.0481977, 13.9713993, -9.0482483, 13.9665203, -23.0147171, 23.0196476
2: -8.0731239, 12.8994694, -8.0719070, 12.8907928, -20.6686172, 20.6870155
3: -9.3757801, 14.5679340, -9.3632097, 14.5725965, -23.9483757, 23.9311447
4: -11.0932055, 13.9776554, -11.1032343, 13.9728889, -25.0402451, 25.0546112
5: -9.1546764, 14.6339579, -9.1496458, 14.6294613, -23.5272903, 23.5333099
6: -20.4385471, 7.4113255, -20.4189281, 7.4128017, -27.8513489, 27.8302536
7: -11.2770748, 16.7762775, -11.2721405, 16.7753162, -27.4860229, 27.4926910
8: -13.3479691, 17.3881912, -13.3572245, 17.3911533, -30.6186218, 30.6339569
9: -6.9307747, 16.0716724, -6.9171829, 16.0754204, -23.0061951, 22.9888554
10: -15.2994766, 19.5809345, -15.2811966, 19.5831509, -34.8826294, 34.8621292
11: -17.7538242, 12.7332726, -17.7481117, 12.7223024, -30.4761276, 30.4813843
12: -21.9741707, 9.5398216, -21.9780788, 9.5614662, -30.0992584, 30.0781631
13: -16.8992119, 14.1532803, -16.8445148, 14.1271286, -31.0263405, 30.9977951
14: -35.5802269, 5.6223488, -35.5709610, 5.5870972, -41.1673241, 41.1933098
15: -14.0603867, 10.5010080, -14.0628633, 10.4887619, -24.5491486, 24.5638714
16: -17.5616512, 14.2035170, -17.5576172, 14.2121181, -31.7737694, 31.7611351
17: -38.8752060, 10.2846737, -38.8811264, 10.2563505, -49.1315575, 49.1658020
18: -19.2643623, 7.6068373, -19.2283897, 7.5261154, -26.7904778, 26.8352280
19: -15.6370630, 3.5393791, -15.6119423, 3.5091300, -19.1461926, 19.1513214
20: -11.3647928, 7.3490043, -11.3686886, 7.3483853, -18.7131786, 18.7176933
21: -17.6957207, 6.7132192, -17.6803627, 6.6899147, -24.3856354, 24.3935814
22: -20.7191067, 6.4121914, -20.7169933, 6.3955145, -27.1146202, 27.1291847
23: -14.2560749, 5.9511499, -14.2433434, 5.9238358, -20.1799107, 20.1944923
24: -17.4989109, 7.4955654, -17.4664211, 7.4355192, -24.9344292, 24.9619865
25: -14.8018780, 7.4058671, -14.7904119, 7.3801646, -22.1820431, 22.1962795
26: -21.2786293, 10.0593624, -21.2623329, 10.0165749, -31.2952042, 31.3216953
27: -17.5226631, 8.2418509, -17.4972191, 8.1902943, -25.7129574, 25.7390709
28: -14.3705978, 7.1104593, -14.3558388, 7.0811892, -21.4517860, 21.4662971
29: -21.8332672, 8.5713825, -21.8590927, 8.5757504, -30.4090176, 30.4304752
30: -16.5101986, 9.7817450, -16.4912224, 9.7490683, -26.2592659, 26.2729683
31: -19.2358360, 5.6619310, -19.2069397, 5.6197371, -24.8555737, 24.8688698
32: -19.1074486, 8.2114277, -19.0721340, 8.1994696, -27.3069191, 27.2835617
33: -33.5153351, 4.6073637, -33.5222321, 4.6318264, -37.6715393, 37.6778107
34: -31.4551888, -0.9378910, -31.4718266, -0.9146404, -29.5356293, 29.5706253
35: -30.2778530, 1.2435589, -30.2921638, 1.2641125, -30.5969849, 30.6154861
36: -27.1433754, 4.0757504, -27.1339474, 4.0847416, -31.1699219, 31.1535568
37: -38.9729156, -2.0145512, -38.9776649, -2.0134697, -36.4726105, 36.5012360
38: -32.1983795, 3.7990026, -32.1991577, 3.7943878, -35.9927673, 35.9981613
39: -37.8004646, 4.4209557, -37.7993431, 4.4426079, -42.1863861, 42.1736450
40: -30.1987514, 4.3690100, -30.1847687, 4.3635006, -34.5622520, 34.5537796
41: -21.4226303, 5.8539076, -21.3845272, 5.8304276, -27.1772919, 27.1650085
42: -12.4935303, 7.0932064, -12.4533768, 7.0755467, -19.5690765, 19.5465832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1705

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1639

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1986961, upper bound: 12.1980208
time: 32.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1986961, upper bound: 12.2017324
time: 47.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.5558739, 10.1049957, -22.6064243, 10.1260586, -32.6819305, 32.7114182
1: -9.0327787, 13.9678164, -9.0586510, 13.9806309, -23.0134087, 23.0264664
2: -8.0688152, 12.9030848, -8.0813065, 12.9081154, -20.6805725, 20.6993752
3: -9.3810873, 14.5634956, -9.3943901, 14.5813532, -23.9624405, 23.9578857
4: -11.0756702, 13.9663944, -11.1173229, 13.9988270, -25.0497055, 25.0630722
5: -9.1533461, 14.6309233, -9.1647224, 14.6439991, -23.5386887, 23.5416412
6: -20.4455643, 7.3924036, -20.4873810, 7.4435716, -27.8891354, 27.8797836
7: -11.2710190, 16.7747841, -11.2862015, 16.7850800, -27.4901047, 27.5049896
8: -13.3361931, 17.3831444, -13.3681946, 17.4098473, -30.6245117, 30.6393433
9: -6.9158063, 16.0328979, -6.9851923, 16.0931969, -23.0090027, 23.0175476
10: -15.2847433, 19.5694962, -15.3118591, 19.5986576, -34.8834000, 34.8813553
11: -17.7312260, 12.7225046, -17.7733192, 12.7659798, -30.4972057, 30.4958229
12: -21.9610558, 9.5104370, -22.0310173, 9.5889893, -30.1396790, 30.1064911
13: -16.9197788, 14.1481152, -16.9409027, 14.1769142, -31.0966930, 31.0890179
14: -35.5648041, 5.6365776, -35.6219635, 5.6960478, -41.2608528, 41.2585411
15: -14.0448265, 10.4973097, -14.0760174, 10.5103626, -24.5551891, 24.5733261
16: -17.5309448, 14.1759472, -17.5887280, 14.2176151, -31.7485600, 31.7646751
17: -38.8151207, 10.2580738, -38.9253998, 10.3900700, -49.2051926, 49.1834717
18: -19.2626076, 7.6456294, -19.2833900, 7.6551266, -26.9177341, 26.9290199
19: -15.6265526, 3.5487070, -15.6369038, 3.5612395, -19.1877918, 19.1856117
20: -11.3416805, 7.3392568, -11.3831139, 7.3786092, -18.7202892, 18.7223701
21: -17.6712837, 6.7078829, -17.7043915, 6.7414660, -24.4127502, 24.4122734
22: -20.6804771, 6.3950958, -20.7437553, 6.4643297, -27.1448059, 27.1388512
23: -14.2462196, 5.9640913, -14.2693768, 5.9786711, -20.2248917, 20.2334671
24: -17.4843388, 7.5186281, -17.5122910, 7.5328932, -25.0172310, 25.0309181
25: -14.7741356, 7.4041176, -14.8137827, 7.4446259, -22.2187614, 22.2178993
26: -21.2640133, 10.0749092, -21.3095970, 10.1067772, -31.3707905, 31.3845062
27: -17.5195103, 8.2644186, -17.5444813, 8.2758236, -25.7953339, 25.8088989
28: -14.3650370, 7.1223574, -14.3808193, 7.1323218, -21.4973583, 21.5031776
29: -21.7827797, 8.5364637, -21.8726845, 8.6402359, -30.4230156, 30.4091492
30: -16.4818745, 9.7722511, -16.5157204, 9.8080692, -26.2899437, 26.2879715
31: -19.2163582, 5.6737099, -19.2437668, 5.6995888, -24.9159470, 24.9174767
32: -19.1291809, 8.2064104, -19.1634388, 8.2402563, -27.3694382, 27.3698502
33: -33.5079460, 4.5992403, -33.5805740, 4.6508875, -37.6857834, 37.7154083
34: -31.4219418, -0.9776535, -31.5022011, -0.9176331, -29.4995270, 29.5602417
35: -30.2548485, 1.2158852, -30.3216228, 1.2644978, -30.5722275, 30.6198120
36: -27.1450253, 4.0727768, -27.1672001, 4.0964146, -31.1861877, 31.1823349
37: -38.9503670, -2.0242634, -39.0114136, -2.0018215, -36.4782944, 36.5258560
38: -32.1788559, 3.7775307, -32.2230606, 3.8045244, -35.9833794, 36.0005913
39: -37.7746696, 4.3917961, -37.8620758, 4.4502964, -42.1687164, 42.2025909
40: -30.1906986, 4.3499842, -30.2541409, 4.3868275, -34.5775261, 34.6041260
41: -21.4273586, 5.8440800, -21.4556427, 5.8614044, -27.2123413, 27.2261047
42: -12.5014324, 7.0788116, -12.5405531, 7.1194067, -19.6208382, 19.6193657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1705

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1956474, upper bound: 12.1986790
time: 43.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1956474, upper bound: 12.1716574
time: 36.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.5932980, 10.1282558, -22.6091766, 10.1323280, -32.7256241, 32.7374344
1: -9.0519991, 13.9775877, -9.0604582, 13.9829149, -23.0349140, 23.0380459
2: -8.0770922, 12.9062691, -8.0823936, 12.9086962, -20.6897659, 20.7037659
3: -9.3863411, 14.5714264, -9.3948936, 14.5833292, -23.9696693, 23.9663200
4: -11.0998096, 13.9892998, -11.1191282, 14.0058069, -25.0808105, 25.0869064
5: -9.1594934, 14.6386967, -9.1657047, 14.6461706, -23.5486832, 23.5531387
6: -20.4639378, 7.4156590, -20.4921150, 7.4460025, -27.9099407, 27.9077740
7: -11.2820835, 16.7802563, -11.2877893, 16.7860813, -27.5021744, 27.5129471
8: -13.3522043, 17.3967438, -13.3697262, 17.4128838, -30.6440659, 30.6547928
9: -6.9544468, 16.0744858, -6.9861646, 16.1062393, -23.0606861, 23.0585785
10: -15.3109989, 19.5868378, -15.3140202, 19.6026993, -34.9136963, 34.9008560
11: -17.7624321, 12.7489204, -17.7822247, 12.7663937, -30.5288258, 30.5311451
12: -21.9960403, 9.5468159, -22.0412407, 9.5910511, -30.1738739, 30.1527786
13: -16.9339314, 14.1592350, -16.9436798, 14.1783543, -31.1122856, 31.1029148
14: -35.5897217, 5.6625042, -35.6282196, 5.6970568, -41.2867775, 41.2907257
15: -14.0650158, 10.5099115, -14.0778828, 10.5137482, -24.5787640, 24.5877953
16: -17.5720062, 14.2086630, -17.5917549, 14.2266903, -31.7986965, 31.8004189
17: -38.8836632, 10.3329868, -38.9456100, 10.3911734, -49.2748375, 49.2785950
18: -19.2700081, 7.6524091, -19.2855415, 7.6554313, -26.9254398, 26.9379501
19: -15.6413975, 3.5576344, -15.6400566, 3.5613570, -19.2027550, 19.1976910
20: -11.3699970, 7.3601828, -11.3908157, 7.3790932, -18.7490902, 18.7509995
21: -17.7023773, 6.7325010, -17.7121468, 6.7417636, -24.4441414, 24.4446487
22: -20.7252598, 6.4364910, -20.7558670, 6.4652328, -27.1904926, 27.1923580
23: -14.2609253, 5.9711370, -14.2729340, 5.9791479, -20.2400742, 20.2440720
24: -17.5058079, 7.5294704, -17.5174370, 7.5334158, -25.0392227, 25.0469074
25: -14.8072119, 7.4291263, -14.8232450, 7.4452229, -22.2524338, 22.2523708
26: -21.2869339, 10.0915527, -21.3151474, 10.1075306, -31.3944645, 31.4067001
27: -17.5295162, 8.2726383, -17.5467205, 8.2760935, -25.8056107, 25.8193588
28: -14.3756590, 7.1284618, -14.3831587, 7.1325569, -21.5082169, 21.5116196
29: -21.8390656, 8.5946054, -21.8886242, 8.6409206, -30.4799862, 30.4832306
30: -16.5171795, 9.8034410, -16.5252705, 9.8090820, -26.3262615, 26.3287125
31: -19.2424889, 5.6906166, -19.2499008, 5.7001677, -24.9426575, 24.9405174
32: -19.1405010, 8.2169628, -19.1657181, 8.2417393, -27.3822403, 27.3826809
33: -33.5363579, 4.6109467, -33.5836906, 4.6542339, -37.7188339, 37.7298737
34: -31.4662075, -0.9351425, -31.5034523, -0.9051619, -29.5594864, 29.6043015
35: -30.2886066, 1.2460785, -30.3233147, 1.2736893, -30.6194534, 30.6510544
36: -27.1556587, 4.0780439, -27.1685581, 4.0959339, -31.2070465, 31.1936417
37: -38.9860611, -2.0099659, -39.0157547, -1.9975796, -36.5299301, 36.5509491
38: -32.2070656, 3.8040686, -32.2249374, 3.8129606, -36.0200272, 36.0290070
39: -37.8240242, 4.4237175, -37.8654022, 4.4606895, -42.2290497, 42.2377167
40: -30.2241325, 4.3720293, -30.2573280, 4.3932076, -34.6173401, 34.6293564
41: -21.4480667, 5.8596287, -21.4572029, 5.8648744, -27.2372437, 27.2432175
42: -12.5258904, 7.0993986, -12.5468445, 7.1205993, -19.6464901, 19.6462440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1705

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1957774, upper bound: 12.2348973
time: 63.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1957774, upper bound: 12.2047219
time: 264.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 330.79 seconds
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 330.79
Output dim: 9, lower bound: -12.1986961, upper bound: 12.1980208
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 330.79
Output dim: 9, lower bound: -12.1986961, upper bound: 12.2017324
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 330.79
Output dim: 9, lower bound: -12.1956474, upper bound: 12.1986790
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 330.79
Output dim: 9, lower bound: -12.1956474, upper bound: 12.1716574
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 330.79
Output dim: 9, lower bound: -12.1957774, upper bound: 12.2348973
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 330.79
Output dim: 9, lower bound: -12.1957774, upper bound: 12.2047219
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 330.79
Output dim: 9, lower bound: -12.2386704, upper bound: 12.2600370
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 330.79
Output dim: 9, lower bound: -12.2386704, upper bound: 12.2604788

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 58.88 + 1879.63 = 1938.51 seconds
