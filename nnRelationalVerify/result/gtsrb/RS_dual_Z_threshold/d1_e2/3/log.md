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
execution time: IAR + RelationalAnalysis = 2.65 + 53.56 = 56.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -12.2647744, upper bound: 12.2647744

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2575884, upper bound: 12.2563563
time: 128.15 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2563563, upper bound: 12.2575884
time: 40.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 168.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 168.79
Output dim: 9, lower bound: -12.2575884, upper bound: 12.2563563
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 168.79
Output dim: 9, lower bound: -12.2563563, upper bound: 12.2575884

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7231445, 20.7231445
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1411896, 25.1416473
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5835114, 23.5839157
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5391006, 27.5395050
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.7004471, 30.7004318
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2593002, 30.2579651
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8119278, 37.8120880
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6452942, 29.6453094
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6890640, 30.6890106
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2388000, 31.2380219
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6015625, 36.5998001
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3137360, 42.3137665
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2781372, 27.2780914
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2564619, upper bound: 12.2551327
time: 59.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2563633, upper bound: 12.2552314
time: 44.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7231445, 20.7231445
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1413574, 25.1411896
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5836639, 23.5835114
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5392303, 27.5391006
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.7004318, 30.7004623
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2579575, 30.2585526
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8120346, 37.8119354
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6454086, 29.6452942
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6890106, 30.6890564
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2380066, 31.2380600
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5998077, 36.6004257
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3137512, 42.3137360
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2780914, 27.2781067
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2552314, upper bound: 12.2563633
time: 51.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2551327, upper bound: 12.2564619
time: 37.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 90.87 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 90.87
Output dim: 9, lower bound: -12.2564619, upper bound: 12.2551327
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 90.87
Output dim: 9, lower bound: -12.2563633, upper bound: 12.2552314
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 90.87
Output dim: 9, lower bound: -12.2552314, upper bound: 12.2563633
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 90.87
Output dim: 9, lower bound: -12.2551327, upper bound: 12.2564619

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7175064, 20.7164307
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1330643, 25.1308517
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5738220, 23.5713501
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5267410, 27.5232010
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6977463, 30.6977081
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2230606, 30.2303314
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8175049, 37.8175278
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6615524, 29.6634521
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7086182, 30.7135925
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2220459, 31.2255707
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6289215, 36.6381531
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3165588, 42.3166046
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2762604, 27.2771301
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2556998, upper bound: 12.2240362
time: 59.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2253645, upper bound: 12.2543715
time: 54.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7164307, 20.7175064
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1304016, 25.1335144
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5709457, 23.5742264
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5227966, 27.5271454
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6977158, 30.6977310
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2316666, 30.2217102
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8173676, 37.8176575
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6634293, 29.6615753
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7136459, 30.7085571
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2263489, 31.2212677
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6399078, 36.6271667
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3165741, 42.3165894
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2771683, 27.2762146
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2556011, upper bound: 12.2241355
time: 47.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2252650, upper bound: 12.2544702
time: 59.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7175064, 20.7164307
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1332397, 25.1304016
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5739746, 23.5709457
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5268707, 27.5227890
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6977310, 30.6977386
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2217178, 30.2309265
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8175964, 37.8173752
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6616592, 29.6634293
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7085571, 30.7136230
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2212677, 31.2256012
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6271667, 36.6387711
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3165894, 42.3165741
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2762146, 27.2771378
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2544701, upper bound: 12.2252650
time: 50.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2241355, upper bound: 12.2556011
time: 31.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7164307, 20.7175064
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1305847, 25.1330643
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5710983, 23.5738220
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5229263, 27.5267410
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6977158, 30.6977615
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2303238, 30.2223053
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8174744, 37.8175049
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6635361, 29.6615524
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7135925, 30.7085876
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2255707, 31.2212906
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6381531, 36.6277924
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3166046, 42.3165588
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2771301, 27.2762222
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2543715, upper bound: 12.2253645
time: 49.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2240362, upper bound: 12.2556998
time: 46.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 98.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.53
Output dim: 9, lower bound: -12.2556998, upper bound: 12.2240362
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.53
Output dim: 9, lower bound: -12.2253645, upper bound: 12.2543715
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.53
Output dim: 9, lower bound: -12.2556011, upper bound: 12.2241355
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.53
Output dim: 9, lower bound: -12.2252650, upper bound: 12.2544702
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.53
Output dim: 9, lower bound: -12.2544701, upper bound: 12.2252650
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.53
Output dim: 9, lower bound: -12.2241355, upper bound: 12.2556011
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.53
Output dim: 9, lower bound: -12.2543715, upper bound: 12.2253645
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.53
Output dim: 9, lower bound: -12.2240362, upper bound: 12.2556998

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7165451, 20.7151947
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1325378, 25.1302414
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5728760, 23.5701447
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5252151, 27.5212402
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6970367, 30.6967621
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2210159, 30.2289047
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8143768, 37.8151398
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6562424, 29.6593246
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7044907, 30.7103806
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2212524, 31.2248993
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6259232, 36.6357193
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3152618, 42.3156281
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2757416, 27.2767410
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2183512, upper bound: 12.2234820
time: 39.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2551459, upper bound: 12.2169148
time: 47.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7162704, 20.7154694
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1324463, 25.1303329
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5726166, 23.5704041
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5247726, 27.5216751
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6968079, 30.6969833
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2216263, 30.2283020
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8151093, 37.8143997
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6574326, 29.6581345
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7054062, 30.7094650
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2213745, 31.2247772
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6265030, 36.6351318
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3155670, 42.3153229
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2758713, 27.2766113
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2183512, upper bound: 12.2538176
time: 46.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2248095, upper bound: 12.2472633
time: 32.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7154694, 20.7162704
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1298752, 25.1329041
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5699921, 23.5730286
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5212708, 27.5251846
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6970062, 30.6967926
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2296524, 30.2202759
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8142548, 37.8152618
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6581192, 29.6574478
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7095261, 30.7053528
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2255554, 31.2205963
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6368790, 36.6247330
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3152771, 42.3156128
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2766495, 27.2758331
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2484952, upper bound: 12.2235806
time: 44.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2484952, upper bound: 12.2171208
time: 43.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7151947, 20.7165451
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1297913, 25.1329880
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5697403, 23.5732803
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5208282, 27.5256195
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6967926, 30.6970062
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2302475, 30.2196808
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8149872, 37.8145294
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6593018, 29.6562653
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7104416, 30.7044296
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2256775, 31.2204742
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6374893, 36.6241531
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3155975, 42.3152924
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2767868, 27.2756958
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2181462, upper bound: 12.2539163
time: 45.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2247108, upper bound: 12.2474696
time: 55.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7165451, 20.7151909
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1327057, 25.1297913
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5730286, 23.5697403
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5253372, 27.5208282
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6970062, 30.6967926
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2196732, 30.2294998
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8144531, 37.8149872
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6563568, 29.6593018
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7044296, 30.7104263
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2204742, 31.2249298
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6241531, 36.6363449
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3152924, 42.3155975
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2756958, 27.2767563
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2474696, upper bound: 12.2247108
time: 47.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2539163, upper bound: 12.2181462
time: 41.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7162704, 20.7154655
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1326218, 25.1298752
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5727692, 23.5699921
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5248947, 27.5212708
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6967926, 30.6970139
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2202835, 30.2288971
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8151855, 37.8142471
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6575470, 29.6581192
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7053452, 30.7095108
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2205963, 31.2248077
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6247330, 36.6357651
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3156128, 42.3152924
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2758331, 27.2766266
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2171208, upper bound: 12.2550473
time: 50.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2235806, upper bound: 12.2484952
time: 47.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7154694, 20.7162666
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1300507, 25.1324463
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5701447, 23.5726166
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5213928, 27.5247726
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6969757, 30.6968155
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2283096, 30.2208786
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8143158, 37.8151093
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6582336, 29.6574326
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7094650, 30.7053909
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2247772, 31.2206268
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6351395, 36.6253662
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3153229, 42.3155823
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2766113, 27.2758484
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2472633, upper bound: 12.2248095
time: 87.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2234820, upper bound: 12.2183512
time: 42.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7151947, 20.7165413
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1299591, 25.1325378
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5698929, 23.5728760
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5209579, 27.5252151
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6967621, 30.6970367
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2289047, 30.2202759
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8150482, 37.8143768
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6594238, 29.6562424
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7103806, 30.7044754
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2248993, 31.2205048
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6357193, 36.6247787
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3156281, 42.3152618
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2767410, 27.2757111
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2169148, upper bound: 12.2551459
time: 47.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2234820, upper bound: 12.2487016
time: 41.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 91.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2183512, upper bound: 12.2234820
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2551459, upper bound: 12.2169148
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2183512, upper bound: 12.2538176
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2248095, upper bound: 12.2472633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2484952, upper bound: 12.2235806
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2484952, upper bound: 12.2171208
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2181462, upper bound: 12.2539163
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2247108, upper bound: 12.2474696
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2474696, upper bound: 12.2247108
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2539163, upper bound: 12.2181462
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2171208, upper bound: 12.2550473
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2235806, upper bound: 12.2484952
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2472633, upper bound: 12.2248095
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2234820, upper bound: 12.2183512
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2169148, upper bound: 12.2551459
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 91.61
Output dim: 9, lower bound: -12.2234820, upper bound: 12.2487016

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7141342, 20.7130547
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1292496, 25.1262283
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5710983, 23.5677032
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5281296, 27.5235519
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6947327, 30.6947632
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2064667, 30.2164154
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8191528, 37.8189621
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6664658, 29.6681290
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7146072, 30.7204590
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2205429, 31.2254257
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6329117, 36.6455002
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3173065, 42.3172607
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2751541, 27.2762146
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 571

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2439082, upper bound: 12.2166698
time: 47.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2549015, upper bound: 12.2056777
time: 52.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7141266, 20.7130470
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1284332, 25.1270447
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5701752, 23.5686188
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5270920, 27.5245972
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6948090, 30.6946869
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2091217, 30.2137527
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8189392, 37.8191681
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6658859, 29.6683578
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7153625, 30.7195816
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2219009, 31.2240677
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6362686, 36.6421280
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3171997, 42.3173676
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2753372, 27.2760239
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 571

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2071125, upper bound: 12.2535732
time: 47.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2181055, upper bound: 12.2425809
time: 33.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7130508, 20.7141342
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1257706, 25.1296997
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5672989, 23.5715027
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5231476, 27.5285416
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6947937, 30.6947098
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2177582, 30.2051315
13: -16.9612427, 14.1880903, -16.9612427, 14.1880903, -31.1493340, 31.1493340
14: -35.6565132, 5.7060785, -35.6565132, 5.7060785, -41.3625908, 41.3625908
15: -14.0987835, 10.5247755, -14.0987835, 10.5247755, -24.6235580, 24.6235580
16: -17.6073990, 14.2553978, -17.6073990, 14.2553978, -31.8627968, 31.8627968
17: -38.9884109, 10.4103050, -38.9884109, 10.4103050, -49.3987160, 49.3987160
18: -19.2994499, 7.6775780, -19.2994499, 7.6775780, -26.9770279, 26.9770279
19: -15.6494017, 3.5682878, -15.6494017, 3.5682878, -19.2176895, 19.2176895
20: -11.4059162, 7.3843813, -11.4059162, 7.3843813, -18.7902985, 18.7902985
21: -17.7248325, 6.7492847, -17.7248325, 6.7492847, -24.4741173, 24.4741173
22: -20.7794247, 6.4735756, -20.7794247, 6.4735756, -27.2530003, 27.2530003
23: -14.2838135, 5.9866066, -14.2838135, 5.9866066, -20.2704201, 20.2704201
24: -17.5296383, 7.5393457, -17.5296383, 7.5393457, -25.0689850, 25.0689850
25: -14.8393555, 7.4546046, -14.8393555, 7.4546046, -22.2939606, 22.2939606
26: -21.3372383, 10.1154041, -21.3372383, 10.1154041, -31.4526424, 31.4526424
27: -17.5600357, 8.2892694, -17.5600357, 8.2892694, -25.8493042, 25.8493042
28: -14.3909273, 7.1405807, -14.3909273, 7.1405807, -21.5315075, 21.5315075
29: -21.9230289, 8.6496887, -21.9230289, 8.6496887, -30.5727177, 30.5727177
30: -16.5374584, 9.8196983, -16.5374584, 9.8196983, -26.3571568, 26.3571568
31: -19.2601662, 5.7072577, -19.2601662, 5.7072577, -24.9674244, 24.9674244
32: -19.1886902, 8.2519283, -19.1886902, 8.2519283, -27.4406185, 27.4406185
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8188019, 37.8192978
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6681061, 29.6664886
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7205124, 30.7145462
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2262039, 31.2197647
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6472549, 36.6311493
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3172302, 42.3173523
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2762527, 27.2751160
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 571

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2069067, upper bound: 12.2536719
time: 44.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2179012, upper bound: 12.2426792
time: 37.79 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 84.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 84.75
Output dim: 9, lower bound: -12.2439082, upper bound: 12.2166698
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 84.75
Output dim: 9, lower bound: -12.2549015, upper bound: 12.2056777
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 84.75
Output dim: 9, lower bound: -12.2071125, upper bound: 12.2535732
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 84.75
Output dim: 9, lower bound: -12.2181055, upper bound: 12.2425809
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 84.75
Output dim: 9, lower bound: -12.2069067, upper bound: 12.2536719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 84.75
Output dim: 9, lower bound: -12.2179012, upper bound: 12.2426792
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 84.75
Output dim: 9, lower bound: -12.2539163, upper bound: 12.2181462
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 84.75
Output dim: 9, lower bound: -12.2171208, upper bound: 12.2550473
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 84.75
Output dim: 9, lower bound: -12.2169148, upper bound: 12.2551459

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 56.21 + 1824.47 = 1880.68 seconds
