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
execution time: IAR + RelationalAnalysis = 2.64 + 53.92 = 56.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -12.2647744, upper bound: 12.2647744

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 666

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2627719, upper bound: 12.2413949
time: 43.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2627719, upper bound: 12.2631999
time: 35.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 79.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 79.08
Output dim: 9, lower bound: -12.2627719, upper bound: 12.2413949
IS_A2, status: Status.UNKNOWN, split count: 1, time: 79.08
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

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 633

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2607061, upper bound: 12.2102974
time: 50.75 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2607061, upper bound: 12.2395021
time: 46.52 seconds

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

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 667

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 633

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2311492, upper bound: 12.2611364
time: 50.97 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2611366, upper bound: 12.2611364
time: 44.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 98.15 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 98.15
Output dim: 9, lower bound: -12.2607061, upper bound: 12.2102974
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 98.15
Output dim: 9, lower bound: -12.2607061, upper bound: 12.2395021
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 98.15
Output dim: 9, lower bound: -12.2311492, upper bound: 12.2611364
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 98.15
Output dim: 9, lower bound: -12.2611366, upper bound: 12.2611364

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -22.5256119, 10.0806332, -22.6095810, 10.1285057, -32.6541176, 32.6902161
1: -9.0020275, 13.9413595, -9.0603085, 13.9774208, -22.9794483, 23.0016670
2: -8.0315933, 12.8690338, -8.0845499, 12.9015255, -20.6474304, 20.6685562
3: -9.3597126, 14.5341244, -9.4016552, 14.5753326, -23.9350452, 23.9357796
4: -11.0297375, 13.9158211, -11.1223507, 13.9934368, -25.0021286, 25.0169067
5: -9.1294012, 14.5893631, -9.1699305, 14.6345339, -23.5141678, 23.5093918
6: -20.4580441, 7.3906293, -20.4975357, 7.4486551, -27.9067001, 27.8881645
7: -11.2149353, 16.7202129, -11.2882957, 16.7724609, -27.4338379, 27.4543839
8: -13.2925110, 17.3341980, -13.3730831, 17.4009876, -30.5804749, 30.5952148
9: -6.9085927, 16.0277309, -6.9894042, 16.0956345, -23.0042267, 23.0171356
10: -15.2821398, 19.5597095, -15.3237410, 19.6037426, -34.8858833, 34.8834496
11: -17.6926861, 12.6933451, -17.7691116, 12.7756395, -30.4683266, 30.4624557
12: -21.8879967, 9.4520922, -22.0128937, 9.5950699, -30.0626221, 30.0445251
13: -16.8961697, 14.1352806, -16.9468861, 14.1799335, -31.0761032, 31.0821667
14: -35.4943542, 5.5806694, -35.6072922, 5.7010803, -41.1954346, 41.1879616
15: -14.0261374, 10.4940653, -14.0860043, 10.5175009, -24.5436382, 24.5800705
16: -17.4952621, 14.1430645, -17.5920486, 14.2225389, -31.7178001, 31.7351131
17: -38.7347412, 10.1870680, -38.9071312, 10.4017496, -49.1364899, 49.0942001
18: -19.2639275, 7.6410947, -19.2852268, 7.6725426, -26.9364700, 26.9263210
19: -15.6004419, 3.5341954, -15.6340837, 3.5658932, -19.1663361, 19.1682796
20: -11.3273716, 7.3315582, -11.3833380, 7.3808198, -18.7081909, 18.7148972
21: -17.6343117, 6.6819992, -17.6988297, 6.7458577, -24.3801689, 24.3808289
22: -20.6076813, 6.3352470, -20.7278290, 6.4682722, -27.0759544, 27.0630760
23: -14.2077332, 5.9367671, -14.2614861, 5.9831333, -20.1908665, 20.1982536
24: -17.4610519, 7.4965305, -17.5110435, 7.5356188, -24.9966698, 25.0075741
25: -14.7337542, 7.3660374, -14.8061018, 7.4485383, -22.1822929, 22.1721382
26: -21.1943760, 10.0218220, -21.2949657, 10.1115704, -31.3059464, 31.3167877
27: -17.5168343, 8.2566729, -17.5465355, 8.2850380, -25.8018723, 25.8032074
28: -14.3335419, 7.0986586, -14.3755903, 7.1373501, -21.4708920, 21.4742489
29: -21.6976776, 8.4641857, -21.8532410, 8.6454601, -30.3431377, 30.3174267
30: -16.4530926, 9.7420931, -16.5125217, 9.8136387, -26.2667313, 26.2546158
31: -19.2007332, 5.6601171, -19.2449722, 5.7031202, -24.9038544, 24.9050903
32: -19.1198196, 8.1987362, -19.1685486, 8.2450466, -27.3648663, 27.3672848
33: -33.4862595, 4.5795527, -33.5846710, 4.6493464, -37.6481476, 37.6957703
34: -31.4232178, -0.9811306, -31.5055351, -0.9121742, -29.4958420, 29.5343628
35: -30.2568359, 1.2286072, -30.3251019, 1.2718296, -30.5645752, 30.6085434
36: -27.1424351, 4.0699663, -27.1703320, 4.1016979, -31.1818542, 31.1866379
37: -38.9443436, -2.0218925, -39.0140533, -1.9862709, -36.4567108, 36.5115585
38: -32.1722946, 3.7693434, -32.2269783, 3.8199530, -35.9922485, 35.9963226
39: -37.7435875, 4.3640747, -37.8680573, 4.4451580, -42.1260071, 42.1774521
40: -30.1509209, 4.3094025, -30.2588043, 4.3834801, -34.5344009, 34.5682068
41: -21.4163780, 5.8318424, -21.4589310, 5.8708277, -27.2102051, 27.2153015
42: -12.4863758, 7.0564451, -12.5409603, 7.1246042, -19.6109810, 19.5974045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2592817, upper bound: 12.1780341
time: 39.67 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2593979, upper bound: 12.2079958
time: 64.80 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -22.5935383, 10.1281109, -22.6151810, 10.1421947, -32.7357330, 32.7432938
1: -9.0512695, 13.9779434, -9.0636625, 13.9877520, -23.0390205, 23.0416069
2: -8.0755548, 12.9070635, -8.0868740, 12.9136143, -20.7038116, 20.7006340
3: -9.3868942, 14.5718575, -9.4027367, 14.5870056, -23.9738998, 23.9745941
4: -11.0979347, 13.9894476, -11.1259060, 14.0167866, -25.0935516, 25.0915527
5: -9.1621590, 14.6382046, -9.1720495, 14.6495361, -23.5624542, 23.5539322
6: -20.4686832, 7.4138842, -20.4995670, 7.4527302, -27.9214134, 27.9134521
7: -11.2810440, 16.7797527, -11.2925367, 16.7907791, -27.5184021, 27.5064468
8: -13.3501453, 17.3972263, -13.3752556, 17.4205151, -30.6582794, 30.6534348
9: -6.9563498, 16.0744362, -6.9921665, 16.1103096, -23.0666599, 23.0666027
10: -15.3068361, 19.5877495, -15.3269615, 19.6081085, -34.9149437, 34.9147110
11: -17.7605133, 12.7478619, -17.7894955, 12.7764311, -30.5369453, 30.5373573
12: -21.9954948, 9.5448246, -22.0470772, 9.5984564, -30.1598358, 30.1720810
13: -16.9318180, 14.1597404, -16.9527397, 14.1832752, -31.1150932, 31.1124802
14: -35.5892258, 5.6638231, -35.6345558, 5.7034302, -41.2926559, 41.2983780
15: -14.0601025, 10.5121384, -14.0902185, 10.5206203, -24.5807228, 24.6023560
16: -17.5731888, 14.2061634, -17.5986977, 14.2392979, -31.8124866, 31.8048611
17: -38.8832016, 10.3321915, -38.9529495, 10.4041576, -49.2873611, 49.2851410
18: -19.2701912, 7.6503477, -19.2904205, 7.6711283, -26.9413185, 26.9407692
19: -15.6416473, 3.5576615, -15.6445704, 3.5659978, -19.2076454, 19.2022324
20: -11.3702364, 7.3610840, -11.3947458, 7.3821106, -18.7523460, 18.7558289
21: -17.7023964, 6.7328181, -17.7176323, 6.7467098, -24.4491062, 24.4504509
22: -20.7246723, 6.4363213, -20.7622566, 6.4704838, -27.1951561, 27.1985779
23: -14.2607155, 5.9710293, -14.2766457, 5.9839001, -20.2446156, 20.2476749
24: -17.5066795, 7.5309696, -17.5228806, 7.5373068, -25.0439873, 25.0538502
25: -14.8075352, 7.4295168, -14.8288383, 7.4513807, -22.2589149, 22.2583542
26: -21.2862358, 10.0921564, -21.3213310, 10.1131916, -31.3994274, 31.4134865
27: -17.5301991, 8.2704945, -17.5511093, 8.2845802, -25.8147793, 25.8216038
28: -14.3760233, 7.1288862, -14.3866997, 7.1380825, -21.5141068, 21.5155869
29: -21.8385162, 8.5947018, -21.8953953, 8.6467915, -30.4853077, 30.4900970
30: -16.5166245, 9.8031387, -16.5308704, 9.8156300, -26.3322544, 26.3340092
31: -19.2425690, 5.6907253, -19.2552319, 5.7046118, -24.9471817, 24.9459572
32: -19.1422405, 8.2161922, -19.1735859, 8.2470083, -27.3892479, 27.3897781
33: -33.5367584, 4.6120319, -33.5897675, 4.6579905, -37.7465134, 37.7295837
34: -31.4679260, -0.9348211, -31.5077896, -0.9004774, -29.6084137, 29.5774994
35: -30.2890968, 1.2459488, -30.3280888, 1.2775574, -30.6522903, 30.6298370
36: -27.1569958, 4.0870876, -27.1743870, 4.1039715, -31.2094574, 31.2087708
37: -38.9866219, -2.0128937, -39.0225563, -1.9843445, -36.5385437, 36.5430145
38: -32.2078476, 3.8000393, -32.2309456, 3.8288355, -36.0366821, 36.0309830
39: -37.8243675, 4.4240513, -37.8739929, 4.4646759, -42.2433014, 42.2418823
40: -30.2243538, 4.3691654, -30.2639885, 4.4013290, -34.6256828, 34.6331558
41: -21.4492073, 5.8596807, -21.4625359, 5.8779354, -27.2544403, 27.2462540
42: -12.5280323, 7.0978212, -12.5534391, 7.1271658, -19.6551971, 19.6512604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 634

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2227530, upper bound: 12.2380744
time: 32.79 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2227530, upper bound: 12.2380744
time: 105.72 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.6130676, 10.1306744, -22.5486984, 10.0930939, -32.7061615, 32.6793747
1: -9.0626192, 13.9793024, -9.0152206, 13.9514761, -23.0140953, 22.9945221
2: -8.0860958, 12.9025650, -8.0418568, 12.8754749, -20.6771011, 20.6602325
3: -9.4028730, 14.5805283, -9.3742676, 14.5535851, -23.9564590, 23.9547958
4: -11.1241379, 14.0048313, -11.0560303, 13.9524021, -25.0553131, 25.0348816
5: -9.1700754, 14.6374207, -9.1388073, 14.6024494, -23.5239868, 23.5284882
6: -20.5099792, 7.4514556, -20.5005112, 7.4284415, -27.9384212, 27.9519672
7: -11.2911310, 16.7736034, -11.2269964, 16.7309952, -27.4688721, 27.4472351
8: -13.3746910, 17.4089584, -13.3157902, 17.3637695, -30.6299286, 30.6137619
9: -6.9901037, 16.1119499, -6.9446955, 16.0795002, -23.0696030, 23.0566444
10: -15.3257647, 19.6052914, -15.2980509, 19.5832520, -34.9090157, 34.9033432
11: -17.7788372, 12.7757502, -17.7276001, 12.7183294, -30.4971657, 30.5033493
12: -22.0347557, 9.5988235, -21.9597855, 9.5053844, -30.1063690, 30.1378632
13: -16.9504204, 14.1826534, -16.9165459, 14.1594868, -31.1099072, 31.0991993
14: -35.6253624, 5.7020493, -35.5555496, 5.6206980, -41.2460594, 41.2575989
15: -14.0881386, 10.5193672, -14.0506458, 10.5035229, -24.5916615, 24.5700130
16: -17.5961800, 14.2192535, -17.5244293, 14.1748066, -31.7709866, 31.7436829
17: -38.9375458, 10.4024067, -38.8322563, 10.2556267, -49.1931725, 49.2346649
18: -19.2899704, 7.6721687, -19.2888794, 7.6527271, -26.9426975, 26.9610481
19: -15.6363773, 3.5655272, -15.6043491, 3.5405915, -19.1769695, 19.1698761
20: -11.3921995, 7.3813272, -11.3594322, 7.3522711, -18.7444706, 18.7407589
21: -17.7031403, 6.7461042, -17.6516075, 6.6947231, -24.3978634, 24.3977127
22: -20.7381401, 6.4692030, -20.6553879, 6.3691511, -27.1072922, 27.1245918
23: -14.2664957, 5.9836283, -14.2272625, 5.9486613, -20.2151566, 20.2108917
24: -17.5150242, 7.5365238, -17.4795132, 7.5034409, -25.0184650, 25.0160370
25: -14.8129292, 7.4497328, -14.7602186, 7.3873620, -22.2002907, 22.2099514
26: -21.3063755, 10.1109905, -21.2392616, 10.0421810, -31.3485565, 31.3502522
27: -17.5518398, 8.2855349, -17.5428009, 8.2657413, -25.8175812, 25.8283348
28: -14.3780193, 7.1372337, -14.3453770, 7.1059241, -21.4839439, 21.4826107
29: -21.8764496, 8.6464233, -21.7751846, 8.5152740, -30.3917236, 30.4216080
30: -16.5157108, 9.8148098, -16.4676208, 9.7533941, -26.2691040, 26.2824306
31: -19.2471924, 5.7037568, -19.2141151, 5.6733751, -24.9205666, 24.9178715
32: -19.1806583, 8.2472553, -19.1618690, 8.2290249, -27.4096832, 27.4091244
33: -33.5888710, 4.6699543, -33.5423012, 4.6455574, -37.7661285, 37.7145996
34: -31.5088902, -0.8989573, -31.4653511, -0.9342794, -29.5833893, 29.5416336
35: -30.3284187, 1.2853260, -30.2978516, 1.2723885, -30.6564102, 30.6186142
36: -27.1736984, 4.1030455, -27.1643753, 4.0855675, -31.2026367, 31.2032700
37: -39.0188141, -1.9820766, -38.9844284, -1.9959106, -36.5656738, 36.5138321
38: -32.2295952, 3.8237467, -32.2009048, 3.7912369, -36.0208321, 36.0246506
39: -37.8703156, 4.4632816, -37.7945938, 4.4222679, -42.2379913, 42.1912689
40: -30.2627506, 4.3947086, -30.1934223, 4.3477192, -34.6104698, 34.5881310
41: -21.4612141, 5.8729315, -21.4312897, 5.8437862, -27.2297440, 27.2273941
42: -12.5505724, 7.1263905, -12.5203638, 7.0840969, -19.6346703, 19.6467552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 667

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 634

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1990282, upper bound: 12.2380746
time: 37.90 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2289794, upper bound: 12.2598404
time: 30.80 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.6186714, 10.1443558, -22.6165962, 10.1405468, -32.7592163, 32.7609520
1: -9.0659714, 13.9896278, -9.0644217, 13.9880552, -23.0540276, 23.0540504
2: -8.0883818, 12.9146671, -8.0857697, 12.9135237, -20.7091522, 20.7166023
3: -9.4039555, 14.5921555, -9.4014482, 14.5913401, -23.9952965, 23.9936028
4: -11.1277027, 14.0281630, -11.1241684, 14.0260801, -25.1299896, 25.1262436
5: -9.1721916, 14.6524792, -9.1715469, 14.6513271, -23.5685577, 23.5767365
6: -20.5120125, 7.4555283, -20.5111446, 7.4517393, -27.9637527, 27.9666729
7: -11.2953482, 16.7919254, -11.2930851, 16.7905445, -27.5209579, 27.5317688
8: -13.3768244, 17.4284668, -13.3733158, 17.4268322, -30.6881104, 30.6915588
9: -6.9928555, 16.1266212, -6.9924016, 16.1261864, -23.1190414, 23.1190224
10: -15.3289547, 19.6095982, -15.3227921, 19.6112785, -34.9402313, 34.9323883
11: -17.7992249, 12.7765398, -17.7954350, 12.7727909, -30.5720158, 30.5719757
12: -22.0688438, 9.6022530, -22.0672550, 9.5980444, -30.2338715, 30.2350998
13: -16.9562702, 14.1860046, -16.9521770, 14.1839628, -31.1402321, 31.1381817
14: -35.6526413, 5.7043667, -35.6504059, 5.7038441, -41.3564835, 41.3547745
15: -14.0923758, 10.5224781, -14.0845900, 10.5215712, -24.6139469, 24.6070671
16: -17.6028175, 14.2360649, -17.6023197, 14.2379389, -31.8407555, 31.8383846
17: -38.9833755, 10.4047813, -38.9806824, 10.4007330, -49.3841095, 49.3854637
18: -19.2951679, 7.6707430, -19.2951813, 7.6620584, -26.9572258, 26.9659233
19: -15.6468525, 3.5655956, -15.6455240, 3.5640399, -19.2108917, 19.2111206
20: -11.4036083, 7.3825946, -11.4023266, 7.3817863, -18.7853947, 18.7849216
21: -17.7219658, 6.7469501, -17.7196541, 6.7455158, -24.4674816, 24.4666042
22: -20.7725945, 6.4713755, -20.7723999, 6.4701900, -27.2427845, 27.2437744
23: -14.2816191, 5.9843817, -14.2802677, 5.9829307, -20.2645493, 20.2646484
24: -17.5268612, 7.5382376, -17.5251999, 7.5379076, -25.0647697, 25.0634384
25: -14.8356781, 7.4525867, -14.8340168, 7.4508281, -22.2865067, 22.2866039
26: -21.3327255, 10.1126328, -21.3311329, 10.1124544, -31.4451790, 31.4437656
27: -17.5564251, 8.2850657, -17.5561619, 8.2795925, -25.8360176, 25.8412285
28: -14.3890915, 7.1379585, -14.3878670, 7.1361194, -21.5252113, 21.5258255
29: -21.9186058, 8.6477947, -21.9160233, 8.6456966, -30.5643024, 30.5638180
30: -16.5340557, 9.8167782, -16.5311642, 9.8143597, -26.3484154, 26.3479424
31: -19.2574768, 5.7052531, -19.2559509, 5.7039771, -24.9614544, 24.9612045
32: -19.1856728, 8.2491884, -19.1843033, 8.2464628, -27.4321365, 27.4334908
33: -33.5939178, 4.6785917, -33.5928192, 4.6780548, -37.7999039, 37.8129501
34: -31.5112076, -0.8871956, -31.5100555, -0.8879833, -29.6264114, 29.6541824
35: -30.3314533, 1.2910347, -30.3301144, 1.2897530, -30.6776962, 30.7063675
36: -27.1777515, 4.1052933, -27.1789703, 4.1027079, -31.2247238, 31.2308426
37: -39.0273094, -1.9801464, -39.0265808, -1.9868965, -36.5972519, 36.5954895
38: -32.2335434, 3.8326321, -32.2364502, 3.8219194, -36.0554619, 36.0690842
39: -37.8762321, 4.4827957, -37.8752861, 4.4822159, -42.3024445, 42.3086243
40: -30.2678795, 4.4124928, -30.2668190, 4.4074507, -34.6753311, 34.6793137
41: -21.4648132, 5.8800316, -21.4640961, 5.8715882, -27.2606506, 27.2715530
42: -12.5630646, 7.1289535, -12.5620975, 7.1254363, -19.6885014, 19.6910515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 590

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 634

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2286965, upper bound: 12.2020725
time: 53.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2287864, upper bound: 12.2598400
time: 46.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 102.34 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 102.34
Output dim: 9, lower bound: -12.2592817, upper bound: 12.1780341
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 102.34
Output dim: 9, lower bound: -12.2593979, upper bound: 12.2079958
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 102.34
Output dim: 9, lower bound: -12.2227530, upper bound: 12.2380744
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 102.34
Output dim: 9, lower bound: -12.2227530, upper bound: 12.2380744
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 102.34
Output dim: 9, lower bound: -12.1990282, upper bound: 12.2380746
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 102.34
Output dim: 9, lower bound: -12.2289794, upper bound: 12.2598404
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 102.34
Output dim: 9, lower bound: -12.2286965, upper bound: 12.2020725
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 102.34
Output dim: 9, lower bound: -12.2287864, upper bound: 12.2598400

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -22.4856644, 10.0542488, -22.6058197, 10.1211777, -32.6068420, 32.6600685
1: -8.9816303, 13.9294796, -9.0580893, 13.9745007, -22.9561310, 22.9875679
2: -8.0219555, 12.8638973, -8.0830040, 12.9003258, -20.6358109, 20.6609650
3: -9.3519001, 14.5240850, -9.4002342, 14.5726662, -23.9245663, 23.9243202
4: -11.0035534, 13.8910093, -11.1198349, 13.9858131, -24.9683838, 24.9895477
5: -9.1182833, 14.5802917, -9.1672716, 14.6318827, -23.4984055, 23.4947433
6: -20.4332905, 7.3647900, -20.4906120, 7.4452868, -27.8785782, 27.8554020
7: -11.2025204, 16.7132263, -11.2862244, 16.7709694, -27.4195557, 27.4437485
8: -13.2748680, 17.3183289, -13.3709421, 17.3972054, -30.5584030, 30.5763092
9: -6.8663549, 15.9839878, -6.9872313, 16.0818939, -22.9482498, 22.9712181
10: -15.2530451, 19.5378647, -15.3206348, 19.5984898, -34.8515358, 34.8584976
11: -17.6585464, 12.6637363, -17.7592716, 12.7741899, -30.4327354, 30.4230080
12: -21.8507671, 9.4131088, -22.0019417, 9.5920925, -30.0219650, 29.9946518
13: -16.8789177, 14.1213131, -16.9431496, 14.1775513, -31.0564690, 31.0644627
14: -35.4665031, 5.5529690, -35.6000137, 5.6994781, -41.1659813, 41.1529846
15: -14.0019073, 10.4780836, -14.0827246, 10.5129967, -24.5149040, 24.5608082
16: -17.4506893, 14.1048861, -17.5878487, 14.2112055, -31.6618958, 31.6927338
17: -38.6620522, 10.1079655, -38.8856468, 10.3992300, -49.0612831, 48.9936142
18: -19.2544823, 7.6267409, -19.2823753, 7.6696596, -26.9241409, 26.9091167
19: -15.5833759, 3.5228581, -15.6301880, 3.5649538, -19.1483307, 19.1530457
20: -11.2967911, 7.3082242, -11.3748932, 7.3795700, -18.6763611, 18.6831169
21: -17.6002579, 6.6550398, -17.6901016, 6.7447128, -24.3449707, 24.3451424
22: -20.5598412, 6.2924461, -20.7146854, 6.4669094, -27.0267506, 27.0071316
23: -14.1912498, 5.9278393, -14.2573471, 5.9820271, -20.1732769, 20.1851864
24: -17.4364033, 7.4840183, -17.5048027, 7.5345097, -24.9709129, 24.9888210
25: -14.6973562, 7.3388472, -14.7955475, 7.4471841, -22.1445408, 22.1343956
26: -21.1688080, 10.0035133, -21.2884998, 10.1102962, -31.2791042, 31.2920132
27: -17.5043373, 8.2444153, -17.5434341, 8.2833605, -25.7876968, 25.7878494
28: -14.3208666, 7.0897532, -14.3725872, 7.1361804, -21.4570465, 21.4623413
29: -21.6377907, 8.4039516, -21.8361435, 8.6440735, -30.2818642, 30.2400951
30: -16.4148045, 9.7083435, -16.5019894, 9.8117599, -26.2265644, 26.2103329
31: -19.1722412, 5.6413994, -19.2380562, 5.7018900, -24.8741302, 24.8794556
32: -19.1049156, 8.1857357, -19.1651173, 8.2427273, -27.3476429, 27.3508530
33: -33.4557953, 4.5647917, -33.5808907, 4.6449680, -37.6121979, 37.6772614
34: -31.3758183, -1.0262136, -31.5033092, -0.9253826, -29.4323730, 29.4847031
35: -30.2208595, 1.1969490, -30.3226490, 1.2619534, -30.5160065, 30.5733719
36: -27.1306534, 4.0523729, -27.1683655, 4.0978794, -31.1608124, 31.1636887
37: -38.9082947, -2.0424042, -39.0089226, -1.9926338, -36.4051208, 36.4824677
38: -32.1447487, 3.7350950, -32.2242355, 3.8093500, -35.9540977, 35.9593315
39: -37.6917915, 4.3293247, -37.8639488, 4.4338264, -42.0624542, 42.1385498
40: -30.1154919, 4.2846022, -30.2549934, 4.3759480, -34.4914398, 34.5395966
41: -21.3954487, 5.8073540, -21.4566002, 5.8643322, -27.1821899, 27.1882172
42: -12.4590359, 7.0335879, -12.5336361, 7.1226468, -19.5816822, 19.5672245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2559392, upper bound: 12.1448898
time: 42.35 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2579350, upper bound: 12.1766488
time: 35.69 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -22.5226059, 10.0774899, -22.6085873, 10.1273994, -32.6500053, 32.6860771
1: -9.0008373, 13.9391651, -9.0599003, 13.9766808, -22.9775181, 22.9990654
2: -8.0301952, 12.8671017, -8.0841026, 12.9008617, -20.6449280, 20.6653099
3: -9.3570404, 14.5319843, -9.4007454, 14.5745907, -23.9316311, 23.9327297
4: -11.0276089, 13.9138775, -11.1216335, 13.9927788, -24.9993515, 25.0133209
5: -9.1243286, 14.5880060, -9.1682310, 14.6340637, -23.5081863, 23.5064011
6: -20.4515190, 7.3878317, -20.4953365, 7.4476900, -27.8992081, 27.8831673
7: -11.2134743, 16.7186584, -11.2878208, 16.7719383, -27.4314957, 27.4516220
8: -13.2908020, 17.3319702, -13.3725052, 17.4002666, -30.5779114, 30.5917740
9: -6.9050345, 16.0255241, -6.9881840, 16.0949116, -22.9999466, 23.0137081
10: -15.2793341, 19.5564079, -15.3227978, 19.6025352, -34.8818703, 34.8792038
11: -17.6897087, 12.6901188, -17.7681255, 12.7745571, -30.4642658, 30.4582443
12: -21.8856411, 9.4494858, -22.0121250, 9.5941734, -30.0561066, 30.0408630
13: -16.8929825, 14.1323347, -16.9458160, 14.1789198, -31.0719032, 31.0781517
14: -35.4913330, 5.5788145, -35.6062355, 5.7004929, -41.1918259, 41.1850510
15: -14.0220985, 10.4905987, -14.0846186, 10.5163536, -24.5384521, 24.5752182
16: -17.4917488, 14.1360207, -17.5908527, 14.2202644, -31.7120132, 31.7268734
17: -38.7306290, 10.1828461, -38.9057693, 10.4003286, -49.1309586, 49.0886154
18: -19.2617378, 7.6334343, -19.2844849, 7.6699429, -26.9316807, 26.9179192
19: -15.5980911, 3.5317485, -15.6332893, 3.5650759, -19.1631660, 19.1650372
20: -11.3250542, 7.3291535, -11.3825779, 7.3800287, -18.7050819, 18.7117310
21: -17.6312294, 6.6796088, -17.6977940, 6.7450638, -24.3762932, 24.3774033
22: -20.6045914, 6.3338146, -20.7267780, 6.4678154, -27.0724068, 27.0605927
23: -14.2059097, 5.9349060, -14.2608824, 5.9825158, -20.1884251, 20.1957893
24: -17.4577484, 7.4948139, -17.5098953, 7.5350285, -24.9927769, 25.0047092
25: -14.7303791, 7.3636880, -14.8048801, 7.4477506, -22.1781292, 22.1685677
26: -21.1916313, 10.0202637, -21.2940598, 10.1110458, -31.3026772, 31.3143234
27: -17.5142632, 8.2526035, -17.5456619, 8.2836456, -25.7979088, 25.7982655
28: -14.3314018, 7.0958457, -14.3748798, 7.1364126, -21.4678154, 21.4707260
29: -21.6940689, 8.4620485, -21.8520470, 8.6447544, -30.3388233, 30.3140945
30: -16.4501152, 9.7394867, -16.5115089, 9.8127737, -26.2628899, 26.2509956
31: -19.1983700, 5.6582851, -19.2441483, 5.7025084, -24.9008789, 24.9024334
32: -19.1161499, 8.1961823, -19.1673450, 8.2441750, -27.3603249, 27.3635273
33: -33.4840088, 4.5765133, -33.5839233, 4.6483412, -37.6450043, 37.6916122
34: -31.4200726, -0.9838409, -31.5044918, -0.9130783, -29.4922028, 29.5282974
35: -30.2544823, 1.2258205, -30.3243275, 1.2709904, -30.5630646, 30.6045609
36: -27.1405888, 4.0576477, -27.1697254, 4.0973587, -31.1764526, 31.1748505
37: -38.9420166, -2.0294838, -39.0132751, -1.9888058, -36.4556046, 36.5053635
38: -32.1697540, 3.7611275, -32.2261009, 3.8170176, -35.9867706, 35.9872284
39: -37.7411385, 4.3612137, -37.8672371, 4.4441853, -42.1227417, 42.1736145
40: -30.1489029, 4.3056164, -30.2581291, 4.3822956, -34.5311966, 34.5637436
41: -21.4140434, 5.8228130, -21.4581528, 5.8678007, -27.2048492, 27.2052383
42: -12.4822140, 7.0540495, -12.5396147, 7.1238098, -19.6060238, 19.5936642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A1_A1_A2_A1

### Relational analysis result of IS_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2258817, upper bound: 12.2050012
time: 27.34 seconds

## Relational analysis of IS_A1_A1_A2_A2

### Relational analysis result of IS_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2580508, upper bound: 12.2065847
time: 35.90 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -22.6120720, 10.1295528, -22.5457001, 10.0898895, -32.7019615, 32.6752548
1: -9.0622082, 13.9785671, -9.0140133, 13.9492645, -23.0114727, 22.9925804
2: -8.0856314, 12.9019241, -8.0404596, 12.8735247, -20.6738434, 20.6576881
3: -9.4019814, 14.5797930, -9.3715572, 14.5514450, -23.9534264, 23.9513512
4: -11.1234245, 14.0041752, -11.0538845, 13.9504623, -25.0517426, 25.0321045
5: -9.1683884, 14.6369715, -9.1337175, 14.6010914, -23.5209198, 23.5224991
6: -20.5078087, 7.4505048, -20.4940109, 7.4256535, -27.9334621, 27.9445152
7: -11.2906494, 16.7730713, -11.2255421, 16.7294388, -27.4661102, 27.4448547
8: -13.3741112, 17.4082298, -13.3140202, 17.3615551, -30.6264877, 30.6111908
9: -6.9888601, 16.1112328, -6.9411006, 16.0773010, -23.0661621, 23.0523338
10: -15.3247986, 19.6040192, -15.2952394, 19.5799332, -34.9047318, 34.8992577
11: -17.7778244, 12.7746582, -17.7246056, 12.7150717, -30.4928970, 30.4992638
12: -22.0339775, 9.5979700, -21.9574242, 9.5027599, -30.1027069, 30.1313705
13: -16.9493942, 14.1816769, -16.9134026, 14.1565351, -31.1059303, 31.0950794
14: -35.6243095, 5.7014265, -35.5524712, 5.6189365, -41.2432480, 41.2538986
15: -14.0867538, 10.5182085, -14.0465822, 10.5000324, -24.5867863, 24.5647907
16: -17.5949821, 14.2170067, -17.5209312, 14.1677818, -31.7627640, 31.7379379
17: -38.9362030, 10.4009438, -38.8280945, 10.2513657, -49.1875687, 49.2290382
18: -19.2892342, 7.6695065, -19.2867050, 7.6450434, -26.9342766, 26.9562111
19: -15.6355896, 3.5647020, -15.6020145, 3.5381498, -19.1737404, 19.1667175
20: -11.3914433, 7.3805065, -11.3571091, 7.3498797, -18.7413235, 18.7376156
21: -17.7021236, 6.7453156, -17.6485291, 6.6923246, -24.3944473, 24.3938446
22: -20.7371025, 6.4687462, -20.6522732, 6.3677368, -27.1048393, 27.1210194
23: -14.2658653, 5.9830055, -14.2254066, 5.9468350, -20.2126999, 20.2084122
24: -17.5138741, 7.5359564, -17.4762344, 7.5017319, -25.0156059, 25.0121918
25: -14.8117294, 7.4489636, -14.7568645, 7.3850360, -22.1967659, 22.2058277
26: -21.3054314, 10.1104336, -21.2365017, 10.0405378, -31.3459702, 31.3469353
27: -17.5509567, 8.2841539, -17.5402069, 8.2616596, -25.8126163, 25.8243599
28: -14.3772659, 7.1362896, -14.3432426, 7.1031303, -21.4803963, 21.4795322
29: -21.8752365, 8.6457348, -21.7715969, 8.5131092, -30.3883457, 30.4173317
30: -16.5146980, 9.8139372, -16.4646358, 9.7507668, -26.2654648, 26.2785721
31: -19.2463799, 5.7031274, -19.2117290, 5.6715641, -24.9179440, 24.9148560
32: -19.1794186, 8.2463551, -19.1581726, 8.2264786, -27.4058971, 27.4045277
33: -33.5880928, 4.6689005, -33.5399742, 4.6425056, -37.7620316, 37.7114105
34: -31.5078945, -0.8998871, -31.4622078, -0.9370012, -29.5773087, 29.5380020
35: -30.3276501, 1.2844658, -30.2955151, 1.2696095, -30.6524429, 30.6170883
36: -27.1730404, 4.0987148, -27.1624985, 4.0732651, -31.1908798, 31.1979370
37: -39.0180473, -1.9846191, -38.9821320, -2.0034637, -36.5595245, 36.5127029
38: -32.2287254, 3.8208351, -32.1983871, 3.7830362, -36.0117607, 36.0192223
39: -37.8694649, 4.4623203, -37.7920799, 4.4193878, -42.2341003, 42.1880112
40: -30.2620392, 4.3935261, -30.1913986, 4.3439102, -34.6059494, 34.5849228
41: -21.4604416, 5.8698711, -21.4289684, 5.8347416, -27.2196503, 27.2220612
42: -12.5492477, 7.1256142, -12.5162201, 7.0816898, -19.6309376, 19.6418343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 590

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.1959702, upper bound: 12.2564950
time: 47.53 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.1661068, upper bound: 12.2584995
time: 34.80 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.6157074, 10.1412182, -22.6155853, 10.1394749, -32.7551804, 32.7568054
1: -9.0648117, 13.9875660, -9.0640335, 13.9874077, -23.0522194, 23.0515995
2: -8.0870361, 12.9127159, -8.0853310, 12.9128637, -20.7066193, 20.7133713
3: -9.4012890, 14.5900326, -9.4005814, 14.5906162, -23.9919052, 23.9906139
4: -11.1256332, 14.0262375, -11.1234646, 14.0254154, -25.1272507, 25.1226883
5: -9.1671562, 14.6511383, -9.1698999, 14.6508713, -23.5626526, 23.5737228
6: -20.5055809, 7.4527874, -20.5089893, 7.4507980, -27.9563789, 27.9617767
7: -11.2939386, 16.7903957, -11.2925930, 16.7900352, -27.5187073, 27.5290375
8: -13.3751183, 17.4262524, -13.3727627, 17.4260864, -30.6856079, 30.6881027
9: -6.9893174, 16.1244278, -6.9911814, 16.1254787, -23.1147957, 23.1156082
10: -15.3261490, 19.6059608, -15.3218460, 19.6100616, -34.9362106, 34.9278069
11: -17.7962303, 12.7733984, -17.7944412, 12.7717228, -30.5679531, 30.5678406
12: -22.0665989, 9.5996552, -22.0664768, 9.5971813, -30.2273788, 30.2314987
13: -16.9532166, 14.1831207, -16.9511795, 14.1829767, -31.1361923, 31.1343002
14: -35.6496277, 5.7025518, -35.6493874, 5.7032528, -41.3528824, 41.3519402
15: -14.0883198, 10.5190430, -14.0832052, 10.5204182, -24.6087379, 24.6022491
16: -17.5993233, 14.2293110, -17.6011276, 14.2357140, -31.8350372, 31.8304386
17: -38.9792480, 10.4005928, -38.9792786, 10.3992748, -49.3785248, 49.3798714
18: -19.2930183, 7.6631341, -19.2944794, 7.6594677, -26.9524860, 26.9576130
19: -15.6445637, 3.5631869, -15.6447582, 3.5632391, -19.2078018, 19.2079449
20: -11.4013119, 7.3802242, -11.4015446, 7.3809681, -18.7822800, 18.7817688
21: -17.7189522, 6.7446246, -17.7186584, 6.7447243, -24.4636765, 24.4632835
22: -20.7694874, 6.4699821, -20.7713661, 6.4696865, -27.2391739, 27.2413483
23: -14.2798071, 5.9825449, -14.2796764, 5.9822969, -20.2621040, 20.2622223
24: -17.5236092, 7.5365477, -17.5240917, 7.5373259, -25.0609360, 25.0606384
25: -14.8323631, 7.4503307, -14.8328581, 7.4500561, -22.2824192, 22.2831879
26: -21.3299980, 10.1110659, -21.3302002, 10.1119595, -31.4419575, 31.4412651
27: -17.5538177, 8.2809620, -17.5552998, 8.2782402, -25.8320580, 25.8362617
28: -14.3869705, 7.1352100, -14.3871517, 7.1351814, -21.5221519, 21.5223618
29: -21.9150162, 8.6456776, -21.9148102, 8.6449852, -30.5600014, 30.5604877
30: -16.5310764, 9.8142042, -16.5302029, 9.8134727, -26.3445492, 26.3444061
31: -19.2551270, 5.7034731, -19.2551498, 5.7033629, -24.9584904, 24.9586220
32: -19.1820335, 8.2467060, -19.1830482, 8.2456245, -27.4276581, 27.4297543
33: -33.5917015, 4.6755877, -33.5920944, 4.6770630, -37.7969208, 37.8088760
34: -31.5080376, -0.8898964, -31.5089951, -0.8889322, -29.6229019, 29.6485825
35: -30.3291817, 1.2887058, -30.3293591, 1.2889385, -30.6762695, 30.7024231
36: -27.1758537, 4.0929852, -27.1783447, 4.0983849, -31.2195587, 31.2192001
37: -39.0250473, -1.9872551, -39.0258179, -1.9890175, -36.5966339, 36.5899658
38: -32.2310791, 3.8246508, -32.2355919, 3.8193884, -36.0504684, 36.0602417
39: -37.8738174, 4.4799404, -37.8745155, 4.4812431, -42.2992249, 42.3047333
40: -30.2658577, 4.4090166, -30.2661152, 4.4063072, -34.6721649, 34.6751328
41: -21.4625359, 5.8710165, -21.4633102, 5.8685317, -27.2553787, 27.2615356
42: -12.5595169, 7.1266317, -12.5608559, 7.1246457, -19.6841621, 19.6874886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 667

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2564952, upper bound: 12.2263084
time: 41.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2584997, upper bound: 12.2584985
time: 36.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 79.99 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 79.99
Output dim: 9, lower bound: -12.2559392, upper bound: 12.1448898
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 79.99
Output dim: 9, lower bound: -12.2579350, upper bound: 12.1766488
IS_A1_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 79.99
Output dim: 9, lower bound: -12.2258817, upper bound: 12.2050012
IS_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 79.99
Output dim: 9, lower bound: -12.2580508, upper bound: 12.2065847
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 79.99
Output dim: 9, lower bound: -12.1959702, upper bound: 12.2564950
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 79.99
Output dim: 9, lower bound: -12.1661068, upper bound: 12.2584995
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 79.99
Output dim: 9, lower bound: -12.2564952, upper bound: 12.2263084
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 79.99
Output dim: 9, lower bound: -12.2584997, upper bound: 12.2584985

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -22.4782887, 10.0422516, -22.5847969, 10.0860081, -32.5642967, 32.6270485
1: -8.9776869, 13.9229298, -9.0451288, 13.9559259, -22.9336128, 22.9680595
2: -8.0178032, 12.8565712, -8.0715942, 12.8797808, -20.6109161, 20.6427650
3: -9.3405724, 14.5204163, -9.3645039, 14.5608768, -23.9014492, 23.8849201
4: -10.9966278, 13.8779621, -11.1021976, 13.9457445, -24.9193268, 24.9546204
5: -9.1130962, 14.5753307, -9.1492262, 14.6142597, -23.4750061, 23.4712448
6: -20.4073219, 7.3602233, -20.4143829, 7.4110198, -27.8183422, 27.7746067
7: -11.1972637, 16.7089081, -11.2691116, 16.7583771, -27.4008255, 27.4213943
8: -13.2704725, 17.3089542, -13.3574877, 17.3711452, -30.5280762, 30.5535507
9: -6.8422213, 15.9809875, -6.9155893, 16.0501766, -22.8923988, 22.8965759
10: -15.2406845, 19.5317688, -15.2836332, 19.5776997, -34.8183823, 34.8154030
11: -17.6497154, 12.6473207, -17.7239876, 12.7261696, -30.3758850, 30.3713074
12: -21.8283463, 9.4057655, -21.9370918, 9.5609035, -29.9551697, 29.9170380
13: -16.8435154, 14.1151905, -16.8416996, 14.1250515, -30.9685669, 30.9568901
14: -35.4567337, 5.5120249, -35.5411911, 5.5850811, -41.0418167, 41.0532150
15: -13.9969921, 10.4685001, -14.0659952, 10.4841728, -24.4811649, 24.5344963
16: -17.4398041, 14.0995598, -17.5508232, 14.1956654, -31.6354694, 31.6503830
17: -38.6533051, 10.0586433, -38.8199654, 10.2588348, -48.9121399, 48.8786087
18: -19.2484646, 7.5805626, -19.2233925, 7.5375381, -26.7860031, 26.8039551
19: -15.5787830, 3.5043886, -15.6006556, 3.5114999, -19.0902824, 19.1050434
20: -11.2914066, 7.2969341, -11.3518553, 7.3482213, -18.6396275, 18.6487885
21: -17.5933456, 6.6354609, -17.6570702, 6.6910696, -24.2844162, 24.2925301
22: -20.5534134, 6.2676716, -20.6743889, 6.3944750, -26.9478874, 26.9420605
23: -14.1862240, 5.9075174, -14.2268972, 5.9248981, -20.1111221, 20.1344147
24: -17.4291878, 7.4494581, -17.4521790, 7.4341469, -24.8633347, 24.9016380
25: -14.6917677, 7.3150616, -14.7615070, 7.3790932, -22.0708618, 22.0765686
26: -21.1602402, 9.9706764, -21.2341652, 10.0158243, -31.1760635, 31.2048416
27: -17.4972229, 8.2134247, -17.4925575, 8.1970100, -25.6942329, 25.7059822
28: -14.3156347, 7.0713220, -14.3443794, 7.0829868, -21.3986206, 21.4157009
29: -21.6317482, 8.3802614, -21.8053493, 8.5762539, -30.2080021, 30.1856117
30: -16.4076576, 9.6861954, -16.4670830, 9.7494593, -26.1571159, 26.1532784
31: -19.1652336, 5.6123872, -19.1931915, 5.6196833, -24.7849159, 24.8055782
32: -19.0710812, 8.1800137, -19.0675049, 8.1994972, -27.2705784, 27.2475185
33: -33.4341431, 4.5610561, -33.5165253, 4.6218472, -37.5636597, 37.6062088
34: -31.3644428, -1.0292416, -31.4700813, -0.9360600, -29.4019623, 29.4446411
35: -30.2097206, 1.1943340, -30.2896709, 1.2519073, -30.4916229, 30.5335007
36: -27.1176567, 4.0500240, -27.1298370, 4.0863438, -31.1284485, 31.1183319
37: -38.8944778, -2.0471592, -38.9672775, -2.0093327, -36.3581467, 36.4277191
38: -32.1355515, 3.7298260, -32.1956291, 3.7897310, -35.9252815, 35.9254532
39: -37.6672821, 4.3264675, -37.7932243, 4.4153137, -42.0184174, 42.0644531
40: -30.0895233, 4.2814846, -30.1789169, 4.3457508, -34.4352722, 34.4604034
41: -21.3694954, 5.8014774, -21.3810749, 5.8290062, -27.1208954, 27.1064911
42: -12.4260464, 7.0271301, -12.4368029, 7.0760512, -19.5020981, 19.4639320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1639

## Relational analysis of IS_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2285133, upper bound: 12.1382889
time: 42.82 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2537513, upper bound: 12.1418526
time: 53.30 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -22.4852962, 10.0532990, -22.6039162, 10.1166992, -32.6019974, 32.6572151
1: -8.9814920, 13.9290991, -9.0573282, 13.9724216, -22.9539146, 22.9864273
2: -8.0217667, 12.8634090, -8.0820532, 12.8977146, -20.6320648, 20.6595154
3: -9.3511257, 14.5238781, -9.3962040, 14.5716343, -23.9227600, 23.9200821
4: -11.0031977, 13.8896084, -11.1180725, 13.9786835, -24.9598160, 24.9869537
5: -9.1179409, 14.5800972, -9.1653109, 14.6309624, -23.4963760, 23.4911346
6: -20.4327431, 7.3645873, -20.4875393, 7.4442225, -27.8769646, 27.8521271
7: -11.2022381, 16.7128811, -11.2847328, 16.7691135, -27.4169922, 27.4416275
8: -13.2746744, 17.3175735, -13.3699799, 17.3928299, -30.5535355, 30.5743866
9: -6.8658724, 15.9838285, -6.9845791, 16.0809727, -22.9468460, 22.9677429
10: -15.2522135, 19.5376339, -15.3164806, 19.5972328, -34.8494453, 34.8541145
11: -17.6583252, 12.6629047, -17.7581043, 12.7702427, -30.4285679, 30.4210091
12: -21.8502750, 9.4128208, -22.0002594, 9.5904751, -30.0298080, 29.9916611
13: -16.8783379, 14.1211090, -16.9408474, 14.1762447, -31.0545826, 31.0619564
14: -35.4661560, 5.5521278, -35.5984344, 5.6950560, -41.1612129, 41.1505623
15: -14.0016155, 10.4773750, -14.0810108, 10.5091810, -24.5107956, 24.5583858
16: -17.4501324, 14.1047134, -17.5849972, 14.2102737, -31.6604061, 31.6897106
17: -38.6617851, 10.1068974, -38.8844833, 10.3936729, -49.0554581, 48.9913788
18: -19.2541199, 7.6261683, -19.2804985, 7.6668029, -26.9209232, 26.9066658
19: -15.5831108, 3.5226192, -15.6287785, 3.5637484, -19.1468582, 19.1513977
20: -11.2966194, 7.3080792, -11.3739500, 7.3789344, -18.6755543, 18.6820297
21: -17.6000271, 6.6546960, -17.6888771, 6.7429314, -24.3429585, 24.3435726
22: -20.5595856, 6.2919559, -20.7132740, 6.4642286, -27.0238152, 27.0052299
23: -14.1910677, 5.9275055, -14.2564716, 5.9802151, -20.1712837, 20.1839771
24: -17.4360847, 7.4833527, -17.5032043, 7.5320587, -24.9681435, 24.9865570
25: -14.6970968, 7.3382659, -14.7943287, 7.4441576, -22.1412544, 22.1325951
26: -21.1684971, 10.0028572, -21.2869568, 10.1067495, -31.2752457, 31.2898140
27: -17.5040607, 8.2442493, -17.5420570, 8.2826681, -25.7867279, 25.7863064
28: -14.3206730, 7.0893192, -14.3716850, 7.1343608, -21.4550343, 21.4610043
29: -21.6375790, 8.4034557, -21.8348827, 8.6414375, -30.2790165, 30.2383385
30: -16.4146461, 9.7078381, -16.5011158, 9.8094721, -26.2241173, 26.2089539
31: -19.1718903, 5.6410809, -19.2361698, 5.7001195, -24.8720093, 24.8772507
32: -19.1041679, 8.1855507, -19.1611004, 8.2418108, -27.3459778, 27.3466511
33: -33.4551239, 4.5646868, -33.5779266, 4.6442699, -37.6108627, 37.6583099
34: -31.3754826, -1.0264359, -31.5016880, -0.9265938, -29.4258652, 29.4782562
35: -30.2204666, 1.1968861, -30.3208008, 1.2614784, -30.5140762, 30.5690384
36: -27.1299210, 4.0523148, -27.1644573, 4.0974808, -31.1654892, 31.1584167
37: -38.9075890, -2.0425606, -39.0054207, -1.9934292, -36.4154358, 36.4774780
38: -32.1442108, 3.7348671, -32.2213974, 3.8082695, -35.9524803, 35.9562645
39: -37.6908226, 4.3292351, -37.8592682, 4.4333668, -42.0610962, 42.1285095
40: -30.1148376, 4.2845345, -30.2514496, 4.3754940, -34.4903336, 34.5359840
41: -21.3949280, 5.8071709, -21.4536972, 5.8634338, -27.1808167, 27.1847076
42: -12.4584427, 7.0332985, -12.5303087, 7.1210995, -19.5795422, 19.5636063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 590

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1639

## Relational analysis of IS_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2285133, upper bound: 12.1382889
time: 52.26 seconds

## Relational analysis of IS_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2285133, upper bound: 12.1382889
time: 44.94 seconds

## BFS IS instance: IS_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -22.5206928, 10.0728531, -22.6082306, 10.1264381, -32.6471329, 32.6810837
1: -9.0000610, 13.9370937, -9.0597687, 13.9763165, -22.9763775, 22.9968624
2: -8.0292635, 12.8644810, -8.0839195, 12.9003487, -20.6434631, 20.6615944
3: -9.3529530, 14.5309057, -9.3999920, 14.5744114, -23.9273643, 23.9308968
4: -11.0258341, 13.9067039, -11.1212921, 13.9913931, -24.9967651, 25.0048141
5: -9.1223679, 14.5871096, -9.1678867, 14.6338768, -23.5045700, 23.5043945
6: -20.4484596, 7.3867679, -20.4948139, 7.4474659, -27.8959255, 27.8815823
7: -11.2119560, 16.7168331, -11.2875357, 16.7716293, -27.4293671, 27.4490509
8: -13.2898235, 17.3276253, -13.3723278, 17.3994541, -30.5759811, 30.5869217
9: -6.9023795, 16.0245953, -6.9876709, 16.0947399, -22.9971199, 23.0122662
10: -15.2751169, 19.5551624, -15.3219614, 19.6022511, -34.8773689, 34.8771248
11: -17.6885452, 12.6862049, -17.7679062, 12.7737370, -30.4622822, 30.4541111
12: -21.8839569, 9.4478340, -22.0116444, 9.5938597, -30.0530853, 30.0486603
13: -16.8907166, 14.1309929, -16.9452362, 14.1786909, -31.0694084, 31.0762291
14: -35.4897537, 5.5743628, -35.6059570, 5.6996784, -41.1894302, 41.1803207
15: -14.0204029, 10.4867830, -14.0843430, 10.5156670, -24.5360699, 24.5711250
16: -17.4888992, 14.1350527, -17.5903034, 14.2200994, -31.7089996, 31.7253571
17: -38.7294617, 10.1772470, -38.9055405, 10.3993034, -49.1287651, 49.0827866
18: -19.2598953, 7.6305509, -19.2841473, 7.6693621, -26.9292564, 26.9146976
19: -15.5966930, 3.5305424, -15.6330357, 3.5648575, -19.1615505, 19.1635780
20: -11.3241215, 7.3285093, -11.3823900, 7.3799143, -18.7040367, 18.7108994
21: -17.6299877, 6.6778388, -17.6975517, 6.7447042, -24.3746910, 24.3753910
22: -20.6031818, 6.3311348, -20.7265415, 6.4673176, -27.0704994, 27.0576763
23: -14.2050018, 5.9330797, -14.2607155, 5.9821701, -20.1871719, 20.1937943
24: -17.4561520, 7.4923058, -17.5095882, 7.5343609, -24.9905128, 25.0018940
25: -14.7291746, 7.3607025, -14.8046503, 7.4472032, -22.1763783, 22.1653519
26: -21.1900826, 10.0167217, -21.2937565, 10.1103964, -31.3004799, 31.3104782
27: -17.5129013, 8.2518702, -17.5453873, 8.2835121, -25.7964134, 25.7972565
28: -14.3305044, 7.0940170, -14.3746777, 7.1359692, -21.4664726, 21.4686947
29: -21.6928062, 8.4594326, -21.8518105, 8.6442375, -30.3370438, 30.3112431
30: -16.4492397, 9.7372065, -16.5113297, 9.8122711, -26.2615108, 26.2485352
31: -19.1964855, 5.6565161, -19.2438145, 5.7021923, -24.8986778, 24.9003296
32: -19.1121235, 8.1952610, -19.1665764, 8.2439852, -27.3561096, 27.3618374
33: -33.4810867, 4.5758152, -33.5832901, 4.6482162, -37.6261139, 37.6902618
34: -31.4184685, -0.9850101, -31.5041046, -0.9133224, -29.4858322, 29.5217285
35: -30.2526360, 1.2253761, -30.3239346, 1.2708817, -30.5587158, 30.6027069
36: -27.1366348, 4.0572605, -27.1689911, 4.0972862, -31.1712341, 31.1794968
37: -38.9384727, -2.0303001, -39.0125427, -1.9889736, -36.4505997, 36.5156937
38: -32.1668396, 3.7600718, -32.2255402, 3.8168344, -35.9836731, 35.9856110
39: -37.7364044, 4.3608074, -37.8662491, 4.4441319, -42.1126862, 42.1722870
40: -30.1453533, 4.3051853, -30.2574844, 4.3822174, -34.5275726, 34.5626678
41: -21.4111786, 5.8219738, -21.4576149, 5.8676271, -27.2013397, 27.2038651
42: -12.4788828, 7.0525026, -12.5389977, 7.1234989, -19.6023827, 19.5914993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 590

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A1_A1_A2_A2_B1

### Relational analysis result of IS_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2516354, upper bound: 12.1794446
time: 300.42 seconds

## Relational analysis of IS_A1_A1_A2_A2_B2

### Relational analysis result of IS_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2559170, upper bound: 12.2043195
time: 42.00 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -22.5910339, 10.0943737, -22.5383148, 10.0778942, -32.6689301, 32.6326904
1: -9.0492325, 13.9600029, -9.0100498, 13.9426975, -22.9919300, 22.9700527
2: -8.0742149, 12.8813877, -8.0363026, 12.8661804, -20.6556473, 20.6327744
3: -9.3661957, 14.5679693, -9.3602524, 14.5477333, -23.9139290, 23.9282227
4: -11.1058064, 13.9641113, -11.0469618, 13.9374323, -25.0168533, 24.9830322
5: -9.1503887, 14.6193295, -9.1285095, 14.5961199, -23.4974899, 23.4991226
6: -20.4315891, 7.4162750, -20.4679794, 7.4211078, -27.8526974, 27.8842545
7: -11.2735128, 16.7604675, -11.2202501, 16.7251091, -27.4437637, 27.4261551
8: -13.3606024, 17.3820953, -13.3096256, 17.3521767, -30.6037292, 30.5808334
9: -6.9172320, 16.0794849, -6.9169459, 16.0742722, -22.9915047, 22.9964314
10: -15.2877693, 19.5833092, -15.2829008, 19.5737953, -34.8615646, 34.8662109
11: -17.7425060, 12.7266388, -17.7157516, 12.6986160, -30.4411221, 30.4423904
12: -21.9691811, 9.5667706, -21.9350777, 9.4953976, -30.0250702, 30.0646286
13: -16.8479996, 14.1291590, -16.8780479, 14.1503201, -30.9983196, 31.0072060
14: -35.5655098, 5.5870399, -35.5427628, 5.5779877, -41.1434975, 41.1298027
15: -14.0700598, 10.4894257, -14.0416412, 10.4904308, -24.5604897, 24.5310669
16: -17.5579681, 14.2014847, -17.5100441, 14.1624298, -31.7203979, 31.7115288
17: -38.8705101, 10.2605686, -38.8194580, 10.2020569, -49.0725670, 49.0800247
18: -19.2302361, 7.5374188, -19.2806644, 7.5988626, -26.8290977, 26.8180828
19: -15.6060648, 3.5112612, -15.5974140, 3.5196991, -19.1257629, 19.1086750
20: -11.3683805, 7.3491640, -11.3517284, 7.3385530, -18.7069340, 18.7008934
21: -17.6691284, 6.6916604, -17.6416092, 6.6727438, -24.3418732, 24.3332691
22: -20.6968079, 6.3962784, -20.6458435, 6.3429537, -27.0397606, 27.0421219
23: -14.2353735, 5.9258842, -14.2203665, 5.9264822, -20.1618557, 20.1462517
24: -17.4612770, 7.4355412, -17.4689865, 7.4671841, -24.9284611, 24.9045277
25: -14.7777090, 7.3808908, -14.7512693, 7.3612523, -22.1389618, 22.1321602
26: -21.2510681, 10.0159264, -21.2279015, 10.0077190, -31.2587872, 31.2438278
27: -17.5000801, 8.1978083, -17.5330544, 8.2306614, -25.7307415, 25.7308617
28: -14.3490458, 7.0830975, -14.3379707, 7.0846624, -21.4337082, 21.4210682
29: -21.8444366, 8.5779724, -21.7655792, 8.4893932, -30.3338299, 30.3435516
30: -16.4797974, 9.7516422, -16.4574223, 9.7286129, -26.2084103, 26.2090645
31: -19.2015209, 5.6209145, -19.2047310, 5.6425304, -24.8440514, 24.8256454
32: -19.0818081, 8.2031441, -19.1243382, 8.2207813, -27.3025894, 27.3274822
33: -33.5236969, 4.6457729, -33.5183258, 4.6388092, -37.6909790, 37.6628265
34: -31.4746265, -0.9105301, -31.4508610, -0.9399967, -29.5372467, 29.5075684
35: -30.2946606, 1.2744637, -30.2844257, 1.2670116, -30.6125488, 30.5927124
36: -27.1345253, 4.0871410, -27.1494789, 4.0708923, -31.1455536, 31.1655960
37: -38.9763870, -2.0013199, -38.9682465, -2.0082188, -36.5048294, 36.4658203
38: -32.2001343, 3.8011560, -32.1891975, 3.7777586, -35.9778938, 35.9903526
39: -37.7986870, 4.4438591, -37.7675781, 4.4165230, -42.1599731, 42.1439590
40: -30.1859436, 4.3633232, -30.1654243, 4.3407879, -34.5267334, 34.5287476
41: -21.3849297, 5.8345928, -21.4030247, 5.8288422, -27.1378632, 27.1607742
42: -12.4524088, 7.0790262, -12.4832211, 7.0752201, -19.5276299, 19.5622482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1705

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1592427, upper bound: 12.2289754
time: 50.97 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1592427, upper bound: 12.2289754
time: 42.90 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -22.6101685, 10.1250706, -22.5453224, 10.0889511, -32.6991196, 32.6703949
1: -9.0614424, 13.9764662, -9.0138741, 13.9489040, -23.0103455, 22.9903412
2: -8.0846958, 12.8993101, -8.0402718, 12.8730297, -20.6723862, 20.6539345
3: -9.3979225, 14.5787354, -9.3708172, 14.5512199, -23.9491425, 23.9495525
4: -11.1216822, 13.9970303, -11.0535593, 13.9490566, -25.0491714, 25.0235291
5: -9.1664352, 14.6360321, -9.1333494, 14.6009274, -23.5173416, 23.5205307
6: -20.5047531, 7.4494262, -20.4934273, 7.4254541, -27.9302063, 27.9428539
7: -11.2891512, 16.7712421, -11.2252464, 16.7291241, -27.4639740, 27.4423218
8: -13.3730965, 17.4038601, -13.3138256, 17.3607750, -30.6245651, 30.6063309
9: -6.9862022, 16.1102943, -6.9406123, 16.0771065, -23.0624542, 23.0509071
10: -15.3206015, 19.6028156, -15.2944365, 19.5796967, -34.9002991, 34.8972511
11: -17.7766647, 12.7707539, -17.7243710, 12.7142563, -30.4909210, 30.4951248
12: -22.0323105, 9.5963402, -21.9569149, 9.5024242, -30.0996628, 30.1391525
13: -16.9471283, 14.1803579, -16.9128456, 14.1563034, -31.1034317, 31.0932045
14: -35.6227493, 5.6970291, -35.5521889, 5.6181059, -41.2408562, 41.2492180
15: -14.0850639, 10.5144110, -14.0462780, 10.4993534, -24.5844173, 24.5606880
16: -17.5921173, 14.2160511, -17.5203781, 14.1676121, -31.7597294, 31.7364292
17: -38.9350433, 10.3954096, -38.8278809, 10.2503862, -49.1854286, 49.2232895
18: -19.2873840, 7.6666555, -19.2863445, 7.6444664, -26.9318504, 26.9529991
19: -15.6341801, 3.5635056, -15.6017332, 3.5379391, -19.1721191, 19.1652393
20: -11.3904991, 7.3798704, -11.3569469, 7.3497605, -18.7402592, 18.7368164
21: -17.7008972, 6.7435217, -17.6483078, 6.6920147, -24.3929119, 24.3918304
22: -20.7356815, 6.4660635, -20.6520081, 6.3672047, -27.1028862, 27.1180725
23: -14.2649746, 5.9811840, -14.2252560, 5.9464684, -20.2114429, 20.2064400
24: -17.5122795, 7.5334444, -17.4759121, 7.5010757, -25.0133553, 25.0093575
25: -14.8105450, 7.4459391, -14.7566319, 7.3844557, -22.1949997, 22.2025719
26: -21.3038673, 10.1069183, -21.2361698, 10.0398808, -31.3437481, 31.3430882
27: -17.5495911, 8.2834702, -17.5399323, 8.2615147, -25.8111057, 25.8234024
28: -14.3763561, 7.1344876, -14.3430786, 7.1026974, -21.4790535, 21.4775658
29: -21.8739929, 8.6430912, -21.7713814, 8.5125980, -30.3865910, 30.4144726
30: -16.5138321, 9.8116646, -16.4644661, 9.7502708, -26.2641029, 26.2761307
31: -19.2444897, 5.7013502, -19.2113800, 5.6712303, -24.9157200, 24.9127312
32: -19.1753979, 8.2454319, -19.1574135, 8.2262669, -27.4016647, 27.4028454
33: -33.5851631, 4.6682110, -33.5393753, 4.6423979, -37.7430420, 37.7101059
34: -31.5062771, -0.9010162, -31.4618778, -0.9372120, -29.5708771, 29.5314789
35: -30.3257904, 1.2840157, -30.2951889, 1.2694988, -30.6481476, 30.6151657
36: -27.1691380, 4.0983353, -27.1617947, 4.0731840, -31.1856079, 31.2025833
37: -39.0144730, -1.9854374, -38.9813957, -2.0036354, -36.5546265, 36.5230484
38: -32.2258835, 3.8196926, -32.1978416, 3.7828445, -36.0087280, 36.0175323
39: -37.8647156, 4.4619255, -37.7911530, 4.4193001, -42.2240295, 42.1866302
40: -30.2585030, 4.3930836, -30.1907558, 4.3438568, -34.6023598, 34.5838394
41: -21.4575653, 5.8690243, -21.4284630, 5.8345509, -27.2160873, 27.2206726
42: -12.5459089, 7.1240816, -12.5156231, 7.0813537, -19.6272621, 19.6397057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 590

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2218624, upper bound: 12.2305823
time: 54.17 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1592427, upper bound: 12.2305823
time: 44.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 101.29 seconds
IS_A1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.2285133, upper bound: 12.1382889
IS_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.2537513, upper bound: 12.1418526
IS_A1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.2285133, upper bound: 12.1382889
IS_A1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.2285133, upper bound: 12.1382889
IS_A1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.2516354, upper bound: 12.1794446
IS_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.2559170, upper bound: 12.2043195
IS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.1592427, upper bound: 12.2289754
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.1592427, upper bound: 12.2289754
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.2218624, upper bound: 12.2305823
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 101.29
Output dim: 9, lower bound: -12.1592427, upper bound: 12.2305823
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 101.29
Output dim: 9, lower bound: -12.2564952, upper bound: 12.2263084
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 101.29
Output dim: 9, lower bound: -12.2584997, upper bound: 12.2584985

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 56.56 + 1748.00 = 1804.56 seconds
