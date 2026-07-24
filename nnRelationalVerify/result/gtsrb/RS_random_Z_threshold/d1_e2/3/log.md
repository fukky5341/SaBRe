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
execution time: IAR + RelationalAnalysis = 2.62 + 53.61 = 56.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -12.2647744, upper bound: 12.2647744

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 733

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2409894, upper bound: 12.2644638
time: 34.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2644638, upper bound: 12.2409894
time: 44.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 79.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 79.36
Output dim: 9, lower bound: -12.2409894, upper bound: 12.2644638
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 79.36
Output dim: 9, lower bound: -12.2644638, upper bound: 12.2409894

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7207336, 20.7213287
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1387939, 25.1394196
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5802155, 23.5810623
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5340729, 27.5353394
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6990356, 30.6993713
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2514496, 30.2491455
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8096008, 37.8086090
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6387329, 29.6364899
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6873474, 30.6851883
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2353363, 31.2343826
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5918732, 36.5890884
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3122559, 42.3117828
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2774811, 27.2771454
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 733

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1616

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2317419, upper bound: 12.2644021
time: 46.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2409273, upper bound: 12.2552146
time: 53.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7213287, 20.7207336
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1394196, 25.1387939
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5810623, 23.5802155
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5353394, 27.5340729
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6993713, 30.6990356
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2491455, 30.2514572
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8086090, 37.8096008
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6364822, 29.6387329
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6851883, 30.6873474
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2343750, 31.2353363
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5890961, 36.5918655
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3117828, 42.3122559
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2771454, 27.2774811
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1641

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1722

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2618562, upper bound: 12.2408747
time: 47.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2643490, upper bound: 12.2383827
time: 47.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 96.61 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 96.61
Output dim: 9, lower bound: -12.2317419, upper bound: 12.2644021
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 96.61
Output dim: 9, lower bound: -12.2409273, upper bound: 12.2552146
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 96.61
Output dim: 9, lower bound: -12.2618562, upper bound: 12.2408747
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 96.61
Output dim: 9, lower bound: -12.2643490, upper bound: 12.2383827

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7217941, 20.7223663
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1390915, 25.1397552
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5806580, 23.5815811
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5354080, 27.5368042
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6999588, 30.7002258
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2531815, 30.2506866
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8124924, 37.8117523
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6433868, 29.6413269
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6907272, 30.6884689
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2356644, 31.2346420
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5938873, 36.5908890
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3134766, 42.3130951
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2779465, 27.2775879
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1568

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2313461, upper bound: 12.2639947
time: 33.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2313372, upper bound: 12.2640036
time: 34.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7217712, 20.7223854
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1391296, 25.1397095
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5807419, 23.5815048
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5355453, 27.5366745
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6998978, 30.7002869
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2529984, 30.2508850
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8127365, 37.8114929
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6435776, 29.6411362
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6906357, 30.6885681
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2356033, 31.2347107
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5936584, 36.5911102
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3135681, 42.3130035
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2779236, 27.2776108
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 633

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2388746, upper bound: 12.2233186
time: 33.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2090425, upper bound: 12.2531659
time: 35.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7210388, 20.7204056
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1393661, 25.1387405
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5808029, 23.5799332
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5349045, 27.5335693
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6991730, 30.6988068
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2484894, 30.2508392
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8079071, 37.8090210
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6351013, 29.6375504
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6842651, 30.6865616
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2342377, 31.2352066
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5883484, 36.5911713
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3116455, 42.3121643
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2770233, 27.2773743
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1642

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2500505, upper bound: 12.2402183
time: 36.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2611953, upper bound: 12.2290621
time: 34.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7210007, 20.7204437
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1393661, 25.1387482
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5807724, 23.5799637
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5348358, 27.5336304
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6991425, 30.6988373
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2485352, 30.2508011
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8080292, 37.8088989
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6352997, 29.6373520
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6843948, 30.6864243
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2342529, 31.2351913
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5883942, 36.5911255
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3116913, 42.3121185
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2770386, 27.2773590
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1648

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 674

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2562116, upper bound: 12.2373289
time: 50.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2632991, upper bound: 12.2302418
time: 46.07 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 98.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.81
Output dim: 9, lower bound: -12.2313461, upper bound: 12.2639947
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.81
Output dim: 9, lower bound: -12.2313372, upper bound: 12.2640036
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 98.81
Output dim: 9, lower bound: -12.2388746, upper bound: 12.2233186
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.81
Output dim: 9, lower bound: -12.2090425, upper bound: 12.2531659
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 98.81
Output dim: 9, lower bound: -12.2500505, upper bound: 12.2402183
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.81
Output dim: 9, lower bound: -12.2611953, upper bound: 12.2290621
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.81
Output dim: 9, lower bound: -12.2562116, upper bound: 12.2373289
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.81
Output dim: 9, lower bound: -12.2632991, upper bound: 12.2302418

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7210999, 20.7216072
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1388550, 25.1394577
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5803146, 23.5811691
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5363541, 27.5376511
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.7002258, 30.7004776
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2506866, 30.2484055
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8132553, 37.8126144
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6450119, 29.6431808
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6922455, 30.6902161
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2363281, 31.2353973
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5957794, 36.5930557
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3138275, 42.3134918
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2781906, 27.2778625
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1641

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2027377, upper bound: 12.2631723
time: 105.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2305235, upper bound: 12.2353812
time: 31.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7210312, 20.7216721
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1387863, 25.1395264
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5802460, 23.5812378
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5362473, 27.5377579
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.7001953, 30.7005081
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2509003, 30.2481918
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8133469, 37.8125153
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6452408, 29.6429520
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6924667, 30.6899872
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2364197, 31.2352905
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5960541, 36.5927887
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3138733, 42.3134460
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2782211, 27.2778320
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 701

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2234266, upper bound: 12.2638665
time: 31.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2312010, upper bound: 12.2561090
time: 49.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7106476, 20.7129135
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1361389, 25.1374664
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5719910, 23.5747910
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5202484, 27.5248337
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6905975, 30.6919785
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2382660, 30.2313843
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8316650, 37.8273392
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6704483, 29.6614609
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.7154388, 30.7067184
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2346039, 31.2330704
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6030579, 36.5956345
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3217010, 42.3197327
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2798080, 27.2784500
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 668

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1623

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.1785517, upper bound: 12.2518241
time: 42.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2076385, upper bound: 12.2227813
time: 38.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7210846, 20.7202950
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1387482, 25.1373062
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5829315, 23.5812149
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5372849, 27.5348206
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6990585, 30.6987152
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2497864, 30.2546768
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8084488, 37.8094406
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6379242, 29.6403732
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6871033, 30.6905289
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2364426, 31.2387314
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5999908, 36.6060715
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3120422, 42.3125305
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2778702, 27.2784119
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 634

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2473343, upper bound: 12.2276981
time: 41.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2486555, upper bound: 12.2263465
time: 49.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7210083, 20.7203445
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1391296, 25.1388092
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5806961, 23.5799713
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5347977, 27.5336227
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6992264, 30.6986618
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2489929, 30.2506638
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8076782, 37.8094940
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6350861, 29.6380844
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6843567, 30.6863937
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2343140, 31.2346725
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5886688, 36.5904465
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3115692, 42.3123779
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2770233, 27.2773438
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 529

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 652

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2473143, upper bound: 12.2369471
time: 60.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2558339, upper bound: 12.2309256
time: 37.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7209015, 20.7204437
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1393661, 25.1385117
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5807724, 23.5798798
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5348358, 27.5335922
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6989670, 30.6988373
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2483978, 30.2508011
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8080292, 37.8085480
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6352997, 29.6371384
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6843948, 30.6863861
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2337341, 31.2351913
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5877228, 36.5911255
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3116913, 42.3119965
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2770386, 27.2773438
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2545862, upper bound: 12.2214813
time: 54.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2545862, upper bound: 12.2214813
time: 54.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 110.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2027377, upper bound: 12.2631723
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2305235, upper bound: 12.2353812
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2234266, upper bound: 12.2638665
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2312010, upper bound: 12.2561090
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.1785517, upper bound: 12.2518241
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2076385, upper bound: 12.2227813
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2473343, upper bound: 12.2276981
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2486555, upper bound: 12.2263465
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2473143, upper bound: 12.2369471
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2558339, upper bound: 12.2309256
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2545862, upper bound: 12.2214813
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 110.78
Output dim: 9, lower bound: -12.2545862, upper bound: 12.2214813

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7190781, 20.7200050
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1358871, 25.1377563
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5755920, 23.5779343
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5329437, 27.5363159
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.7000580, 30.7003098
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1254196, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2563934, 30.2498093
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8134460, 37.8128052
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6478271, 29.6453781
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6976318, 30.6931915
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2415771, 31.2386856
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.6154709, 36.6074295
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3141785, 42.3138428
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2795944, 27.2788010
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 636

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 753

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2019846, upper bound: 12.2571417
time: 52.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.1967077, upper bound: 12.2624217
time: 31.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7198944, 20.7208176
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1382599, 25.1391220
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5796051, 23.5808640
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5356445, 27.5375977
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6993103, 30.6998367
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2489929, 30.2456436
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8158875, 37.8143234
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6435547, 29.6400375
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6931534, 30.6897354
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2362366, 31.2349319
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5951538, 36.5911942
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3150482, 42.3143005
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2778015, 27.2772751
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1648

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2136610, upper bound: 12.2634895
time: 42.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2230437, upper bound: 12.2541148
time: 36.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7201767, 20.7205353
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1383896, 25.1389923
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5798721, 23.5806046
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5360794, 27.5371552
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6995239, 30.6996231
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2483673, 30.2462769
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8151550, 37.8150482
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6423264, 29.6412659
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6922150, 30.6906738
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2360687, 31.2350922
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5944519, 36.5918884
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3147430, 42.3146210
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2776642, 27.2774124
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1600

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2276399, upper bound: 12.2560838
time: 44.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2308782, upper bound: 12.2499958
time: 31.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7200546, 20.7183914
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1386261, 25.1379700
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5798492, 23.5781250
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5332336, 27.5304260
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6988602, 30.6975479
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2452927, 30.2494278
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8029022, 37.8071976
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6267700, 29.6340179
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6781387, 30.6835327
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2329483, 31.2336807
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5835419, 36.5879059
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3096466, 42.3115082
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2760620, 27.2768707
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 590

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1703

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2550881, upper bound: 12.2165289
time: 48.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2389460, upper bound: 12.2301901
time: 183.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.6215534, 10.1469803, -22.6215534, 10.1469803, -32.7685318, 32.7685318
1: -9.0676470, 13.9918318, -9.0676470, 13.9918318, -23.0594788, 23.0594788
2: -8.0905628, 12.9161167, -8.0905628, 12.9161167, -20.7203827, 20.7204514
3: -9.4058743, 14.5945339, -9.4058743, 14.5945339, -24.0004082, 24.0004082
4: -11.1309013, 14.0316277, -11.1309013, 14.0316277, -25.1405945, 25.1384888
5: -9.1762676, 14.6546259, -9.1762676, 14.6546259, -23.5811615, 23.5798645
6: -20.5148926, 7.4591651, -20.5148926, 7.4591651, -27.9740582, 27.9740582
7: -11.2974005, 16.7944145, -11.2974005, 16.7944145, -27.5346222, 27.5335770
8: -13.3794146, 17.4308453, -13.3794146, 17.4308453, -30.6988373, 30.6988220
9: -6.9961228, 16.1304150, -6.9961228, 16.1304150, -23.1265373, 23.1265373
10: -15.3348579, 19.6172047, -15.3348579, 19.6172047, -34.9520645, 34.9520645
11: -17.8033886, 12.7808084, -17.8033886, 12.7808084, -30.5841980, 30.5841980
12: -22.0727139, 9.6056099, -22.0727139, 9.6056099, -30.2482910, 30.2537842
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
33: -33.5969162, 4.6815853, -33.5969162, 4.6815853, -37.8079834, 37.8082886
34: -31.5128002, -0.8836060, -31.5128002, -0.8836060, -29.6351166, 29.6355209
35: -30.3336678, 1.2938995, -30.3336678, 1.2938995, -30.6840363, 30.6839066
36: -27.1817322, 4.1078434, -27.1817322, 4.1078434, -31.2336121, 31.2372513
37: -39.0308990, -1.9708328, -39.0308990, -1.9708328, -36.5874481, 36.5955887
38: -32.2415314, 3.8407001, -32.2415314, 3.8407001, -36.0822296, 36.0822296
39: -37.8806877, 4.4866676, -37.8806877, 4.4866676, -42.3116760, 42.3117981
40: -30.2708626, 4.4184999, -30.2708626, 4.4184999, -34.6893616, 34.6893616
41: -21.4669113, 5.8852654, -21.4669113, 5.8852654, -27.2770081, 27.2769852
42: -12.5656614, 7.1320348, -12.5656614, 7.1320348, -19.6976967, 19.6976967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=264, inp2_unstable=264, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=41, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -12.2508477, upper bound: 12.2211379
time: 54.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -12.2542422, upper bound: 12.2202939
time: 50.00 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 106.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2019846, upper bound: 12.2571417
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.1967077, upper bound: 12.2624217
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2136610, upper bound: 12.2634895
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2230437, upper bound: 12.2541148
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2276399, upper bound: 12.2560838
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2308782, upper bound: 12.2499958
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2550881, upper bound: 12.2165289
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2389460, upper bound: 12.2301901
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2508477, upper bound: 12.2211379
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 106.12
Output dim: 9, lower bound: -12.2542422, upper bound: 12.2202939
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 106.12
Output dim: 9, lower bound: -12.2545862, upper bound: 12.2214813

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 56.22 + 1786.55 = 1842.77 seconds
