## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 1800 seconds
Split limit: 100
Threshold: 5.0897715336


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4349213, 13.4349213)
1: (-6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.4018059, 6.4018040)
2: (-11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5668907, 8.5668926)
3: (-12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1626625, 11.1626587)
4: (-22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2590103, 9.2590103)
5: (-10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1764145, 12.1764145)
6: (-22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1619263, 11.1619263)
7: (-9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4939041, 12.4939041)
8: (-26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8380928, 9.8380928)
9: (-14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9313660, 12.9313698)
10: (-5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2206535, 13.2206535)
11: (9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4874420, 7.4874420)
12: (-15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.5037537, 18.5037537)
13: (-28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8806000, 12.8806000)
14: (-31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4959564, 21.4959564)
15: (-24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8917961, 8.8917942)
16: (-6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1148109, 10.1148090)
17: (-14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7216034, 21.7216034)
18: (-0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6829491, 10.6829491)
19: (-5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5620136, 7.5620117)
20: (-3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2555389, 10.2555389)
21: (-1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9094620, 8.9094620)
22: (-9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6781940, 8.6781921)
23: (1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7109337, 7.7109337)
24: (-2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1370564, 8.1370583)
25: (0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3601608, 9.3601608)
26: (-17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5705528, 14.5705566)
27: (-10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1072502, 9.1072521)
28: (1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6096764, 9.6096764)
29: (-5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6049614, 8.6049614)
30: (5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6533470, 7.6533451)
31: (-3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1470108, 9.1470108)
32: (-19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6920662, 10.6920643)
33: (-47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5447121, 14.5447121)
34: (-29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6652718, 10.6652718)
35: (-29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7034035, 10.7034035)
36: (-31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6921921, 12.6921959)
37: (-46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0821533, 16.0821533)
38: (-34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1194077, 15.1194077)
39: (-56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1927299, 13.1927299)
40: (-40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1979179, 8.1979179)
41: (-26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2451687, 11.2451706)
42: (-14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5134354, 8.5134354)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.77 + 26.63 = 29.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -5.0948664, upper bound: 5.0948664

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1624

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0937876, upper bound: 5.0862820
time: 21.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0937876, upper bound: 5.0937876
time: 20.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 42.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 42.13
Output dim: 11, lower bound: -5.0937876, upper bound: 5.0862820
IS_A2, status: Status.UNKNOWN, split count: 1, time: 42.13
Output dim: 11, lower bound: -5.0937876, upper bound: 5.0937876

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -20.2234631, 0.7040813, -20.2254581, 0.7049952, -13.4290924, 13.4300117
1: -6.4232450, 5.2885194, -6.4237509, 5.2891188, -6.3983917, 6.3981037
2: -11.0243120, 2.2231686, -11.0246077, 2.2240310, -8.5629730, 8.5631733
3: -12.3313494, 3.3935637, -12.3393574, 3.3944726, -11.1430244, 11.1499786
4: -22.0535736, -5.6432390, -22.0539227, -5.6424646, -9.2513046, 9.2533569
5: -10.8011379, 5.6414452, -10.8041801, 5.6421323, -12.1654587, 12.1686935
6: -22.4850388, -4.5092411, -22.4885540, -4.5090580, -11.1548920, 11.1572685
7: -9.5649939, 8.9270668, -9.5665398, 8.9277077, -12.4782410, 12.4831619
8: -26.3497009, -5.6010666, -26.3505154, -5.5995855, -9.8327866, 9.8331451
9: -14.5699644, 2.1377993, -14.5811605, 2.1396201, -12.9015884, 12.9121666
10: -5.8971176, 11.7156019, -5.9000101, 11.7175961, -13.2109032, 13.2121468
11: 9.5995378, 21.1070709, 9.5983744, 21.1130714, -7.4745283, 7.4697018
12: -15.1666822, 9.8661413, -15.1680756, 9.8675852, -18.4892578, 18.4834061
13: -28.0081692, -3.0816476, -28.0205860, -3.0793211, -12.8463593, 12.8568916
14: -31.3538208, 0.6504951, -31.3565521, 0.6597505, -21.4802017, 21.4725037
15: -24.9338493, -10.6949463, -24.9353371, -10.6937733, -8.8879280, 8.8876247
16: -6.9424772, 7.8869638, -6.9458117, 7.8875189, -10.1066704, 10.1091576
17: -14.7388268, 11.7490082, -14.7403603, 11.7626143, -21.7004547, 21.6872559
18: -0.8714225, 12.5548887, -0.8733456, 12.5673580, -10.6628914, 10.6526222
19: -5.2731838, 4.7356901, -5.2740440, 4.7402039, -7.5540123, 7.5510693
20: -3.3994706, 7.9687858, -3.4008954, 7.9693618, -10.2458954, 10.2485962
21: -1.9286330, 8.8783035, -1.9297690, 8.8801556, -8.9029579, 8.9036484
22: -9.2044983, 2.7817874, -9.2051458, 2.7835922, -8.6744576, 8.6732883
23: 1.3818816, 12.5043678, 1.3812000, 12.5087185, -7.7037811, 7.7003975
24: -2.6487770, 10.4851475, -2.6492243, 10.4924965, -8.1260281, 8.1199112
25: 0.3978972, 13.7618446, 0.3970685, 13.7643538, -9.3554382, 9.3536911
26: -17.3828220, 2.4711204, -17.3841400, 2.4782104, -14.5578232, 14.5523376
27: -10.2636986, 6.2768283, -10.2657318, 6.2850008, -9.0924072, 9.0886383
28: 1.1137371, 13.5755920, 1.1127429, 13.5787659, -9.6040611, 9.6025352
29: -5.0823560, 8.3510427, -5.0833015, 8.3557196, -8.5968552, 8.5924988
30: 6.0005770, 17.7140160, 5.9998922, 17.7174377, -7.6463680, 7.6452541
31: -3.3588080, 10.3898954, -3.3599892, 10.3944044, -9.1378326, 9.1366806
32: -19.5758762, -2.7767177, -19.5846786, -2.7758453, -10.6737289, 10.6802025
33: -47.0201187, -21.5766945, -47.0316734, -21.5749321, -14.5181274, 14.5255356
34: -29.7169075, -10.5947285, -29.7226677, -10.5943594, -10.6553116, 10.6572609
35: -29.2184658, -9.9666777, -29.2244682, -9.9659700, -10.6916199, 10.6912804
36: -31.8848782, -9.4220352, -31.8892937, -9.4211607, -12.6836319, 12.6852493
37: -46.1353836, -23.4878883, -46.1375580, -23.4864540, -16.0707245, 16.0672073
38: -34.2988739, -11.5235538, -34.3016586, -11.5228968, -15.1123047, 15.1131554
39: -56.3064728, -30.6862144, -56.3176155, -30.6853180, -13.1636276, 13.1725998
40: -40.2469139, -23.3092613, -40.2522316, -23.3091164, -8.1849823, 8.1895790
41: -26.7420769, -7.0024719, -26.7437687, -7.0020866, -11.2409191, 11.2418861
42: -14.5243530, -2.1243496, -14.5306501, -2.1239743, -8.5023556, 8.5060158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 625

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0835327, upper bound: 5.0859441
time: 23.53 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0835327, upper bound: 5.0860455
time: 18.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -20.2298107, 0.7215319, -20.2273445, 0.7059021, -13.4380951, 13.4496574
1: -6.4298010, 5.2930298, -6.4243188, 5.2892189, -6.4037209, 6.4083843
2: -11.0285702, 2.2373319, -11.0249453, 2.2250304, -8.5611687, 8.5902100
3: -12.3505373, 3.4306407, -12.3493204, 3.3954911, -11.1564713, 11.1962891
4: -22.0554810, -5.6315928, -22.0541496, -5.6414809, -9.2497902, 9.2774086
5: -10.8089733, 5.6646576, -10.8080482, 5.6429043, -12.1717834, 12.2038078
6: -22.4991550, -4.4887033, -22.4926815, -4.5088339, -11.1699295, 11.1810760
7: -9.5707054, 8.9374361, -9.5682564, 8.9284782, -12.4765244, 12.5214348
8: -26.3635616, -5.5972538, -26.3514633, -5.5986924, -9.8464813, 9.8451920
9: -14.5978165, 2.1959472, -14.5951614, 2.1418598, -12.9180984, 12.9878311
10: -5.9042349, 11.7575130, -5.9016857, 11.7200108, -13.2216988, 13.2621498
11: 9.5698137, 21.1193848, 9.5972815, 21.1195488, -7.5131845, 7.4773788
12: -15.1794586, 9.8730783, -15.1696901, 9.8692904, -18.5435181, 18.4763641
13: -28.0350170, -3.0040066, -28.0346336, -3.0767379, -12.8628540, 12.9569473
14: -31.4098377, 0.6739292, -31.3596077, 0.6708221, -21.5462723, 21.4943771
15: -24.9363060, -10.6817369, -24.9363098, -10.6926270, -8.8933563, 8.9030342
16: -6.9511375, 7.9121356, -6.9489765, 7.8882146, -10.1145630, 10.1376705
17: -14.7954168, 11.7787218, -14.7422228, 11.7787380, -21.7806396, 21.7164459
18: -0.9359524, 12.5838518, -0.8754947, 12.5829659, -10.7444077, 10.6670532
19: -5.3054438, 4.7455068, -5.2750320, 4.7452550, -7.5903645, 7.5574303
20: -3.4138196, 7.9742627, -3.4023955, 7.9700351, -10.2501907, 10.2773705
21: -1.9537711, 8.8822136, -1.9310250, 8.8819180, -8.9237633, 8.9224911
22: -9.2310429, 2.7866256, -9.2059174, 2.7853804, -8.7025261, 8.6789894
23: 1.3529805, 12.5129986, 1.3804908, 12.5125742, -7.7399712, 7.7060089
24: -2.6863058, 10.5034933, -2.6496789, 10.5009823, -8.1743450, 8.1314964
25: 0.3755047, 13.7686062, 0.3961167, 13.7667761, -9.3808250, 9.3601456
26: -17.4366474, 2.4875755, -17.3856106, 2.4871616, -14.6190147, 14.5622101
27: -10.3289146, 6.2938704, -10.2681580, 6.2941179, -9.1680794, 9.1074505
28: 1.0730166, 13.5824547, 1.1116111, 13.5821533, -9.6480980, 9.6083336
29: -5.1237993, 8.3616152, -5.0844345, 8.3606853, -8.6464539, 8.6013756
30: 5.9816198, 17.7215385, 5.9991870, 17.7209358, -7.6651192, 7.6560650
31: -3.3927443, 10.3992214, -3.3613448, 10.3994379, -9.1753693, 9.1502647
32: -19.6029434, -2.7406456, -19.5946350, -2.7747812, -10.6946144, 10.7253723
33: -47.0455322, -21.5092182, -47.0445251, -21.5730934, -14.5352516, 14.6118279
34: -29.7334576, -10.5693274, -29.7279816, -10.5939713, -10.6875992, 10.6833725
35: -29.2333221, -9.9370108, -29.2301216, -9.9653368, -10.7220154, 10.7123642
36: -31.9036884, -9.3957710, -31.8936024, -9.4201183, -12.7202187, 12.7086143
37: -46.1442413, -23.4701443, -46.1399879, -23.4847183, -16.1096115, 16.0695877
38: -34.3130836, -11.5094376, -34.3048325, -11.5220852, -15.1465225, 15.1219940
39: -56.3275108, -30.6267662, -56.3277588, -30.6842880, -13.1768723, 13.2539711
40: -40.2589226, -23.2915535, -40.2577248, -23.3089027, -8.1935196, 8.2141247
41: -26.7542706, -6.9855852, -26.7455330, -7.0017133, -11.2656078, 11.2591553
42: -14.5436287, -2.0951018, -14.5382690, -2.1235557, -8.5180206, 8.5359764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 625

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0835327, upper bound: 5.0934496
time: 16.99 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0935512, upper bound: 5.0935513
time: 13.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 32.71 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 32.71
Output dim: 11, lower bound: -5.0835327, upper bound: 5.0859441
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 32.71
Output dim: 11, lower bound: -5.0835327, upper bound: 5.0860455
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.71
Output dim: 11, lower bound: -5.0835327, upper bound: 5.0934496
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.71
Output dim: 11, lower bound: -5.0935512, upper bound: 5.0935513

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -20.2287979, 0.7183027, -20.2255192, 0.6998951, -13.4267769, 13.4427795
1: -6.4294424, 5.2902222, -6.4236579, 5.2840490, -6.3956985, 6.4036598
2: -11.0233908, 2.2369280, -11.0153332, 2.2242210, -8.5549850, 8.5801315
3: -12.3471355, 3.4301624, -12.3430614, 3.3945665, -11.1513519, 11.1888046
4: -22.0508518, -5.6323366, -22.0455551, -5.6428251, -9.2440605, 9.2684708
5: -10.8026514, 5.6639371, -10.7962532, 5.6416135, -12.1632843, 12.1904221
6: -22.4847107, -4.4889765, -22.4659004, -4.5093498, -11.1549416, 11.1538582
7: -9.5684652, 8.9359570, -9.5640917, 8.9257269, -12.4702454, 12.5152321
8: -26.3614616, -5.5977578, -26.3476181, -5.5996528, -9.8400574, 9.8389130
9: -14.5970764, 2.1944222, -14.5937748, 2.1390066, -12.9122620, 12.9829178
10: -5.9032211, 11.7562065, -5.8997869, 11.7175694, -13.2180176, 13.2589302
11: 9.5714321, 21.1097736, 9.6002817, 21.1017342, -7.4938869, 7.4643536
12: -15.1751328, 9.8721561, -15.1618452, 9.8677197, -18.5370255, 18.4664459
13: -28.0339489, -3.0084674, -28.0326538, -3.0850861, -12.8539581, 12.9511757
14: -31.4081440, 0.6678841, -31.3565025, 0.6596258, -21.5299683, 21.4836273
15: -24.9358082, -10.6837044, -24.9353657, -10.6961899, -8.8889236, 8.9000740
16: -6.9500680, 7.9105577, -6.9470539, 7.8853755, -10.1085968, 10.1333313
17: -14.7942638, 11.7747755, -14.7399826, 11.7714920, -21.7684631, 21.7083588
18: -0.9349823, 12.5824070, -0.8736367, 12.5804062, -10.7406006, 10.6632957
19: -5.3045087, 4.7435579, -5.2733288, 4.7416925, -7.5850410, 7.5535507
20: -3.4109406, 7.9737806, -3.3970532, 7.9691796, -10.2455444, 10.2699394
21: -1.9524190, 8.8803806, -1.9285040, 8.8785915, -8.9196434, 8.9185963
22: -9.2301598, 2.7850146, -9.2042665, 2.7824354, -8.6981544, 8.6752968
23: 1.3539163, 12.5060434, 1.3822207, 12.4996567, -7.7263756, 7.6977310
24: -2.6856813, 10.4938908, -2.6484954, 10.4833612, -8.1547966, 8.1203156
25: 0.3763390, 13.7530966, 0.3976929, 13.7380362, -9.3513489, 9.3436203
26: -17.4351749, 2.4870057, -17.3828125, 2.4861112, -14.6161385, 14.5580902
27: -10.3272285, 6.2928648, -10.2649603, 6.2922173, -9.1645565, 9.1023006
28: 1.0741334, 13.5773020, 1.1136906, 13.5726948, -9.6370430, 9.6010551
29: -5.1230927, 8.3598042, -5.0830975, 8.3573742, -8.6422844, 8.5982857
30: 5.9828305, 17.7124786, 6.0014229, 17.7042809, -7.6475677, 7.6446400
31: -3.3914375, 10.3945847, -3.3588898, 10.3909159, -9.1656265, 9.1435509
32: -19.5895233, -2.7411594, -19.5706329, -2.7757285, -10.6798325, 10.6988735
33: -47.0431786, -21.5105438, -47.0401459, -21.5755405, -14.5302010, 14.6062965
34: -29.7294407, -10.5699959, -29.7206001, -10.5951929, -10.6814156, 10.6735077
35: -29.2310276, -9.9379368, -29.2258263, -9.9670410, -10.7167397, 10.7055550
36: -31.8993225, -9.3966637, -31.8855381, -9.4217510, -12.7133827, 12.6978531
37: -46.1422806, -23.4733639, -46.1364288, -23.4903793, -16.0970535, 16.0607986
38: -34.3094330, -11.5101166, -34.2984390, -11.5233002, -15.1411209, 15.1140022
39: -56.3255692, -30.6286240, -56.3240662, -30.6876373, -13.1712723, 13.2478600
40: -40.2524033, -23.2916794, -40.2456474, -23.3091316, -8.1866722, 8.2019463
41: -26.7440300, -6.9861431, -26.7266369, -7.0028157, -11.2541828, 11.2394791
42: -14.5414276, -2.0955672, -14.5342455, -2.1244178, -8.5151329, 8.5316505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0825284, upper bound: 5.0883658
time: 16.77 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0825284, upper bound: 5.0924408
time: 22.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -20.2292404, 0.7162790, -20.2340336, 0.7051268, -13.4417038, 13.4702835
1: -6.4296484, 5.2889528, -6.4306927, 5.2890596, -6.4061184, 6.4240284
2: -11.0277843, 2.2371409, -11.0260067, 2.2459192, -8.5818443, 8.5882225
3: -12.3493767, 3.4303062, -12.3519173, 3.4094353, -11.1700401, 11.1974792
4: -22.0548611, -5.6319566, -22.0549870, -5.6237845, -9.2678528, 9.2750969
5: -10.8080950, 5.6643686, -10.8087587, 5.6684356, -12.1950684, 12.2006035
6: -22.4971352, -4.4889135, -22.4923553, -4.4649019, -11.2120056, 11.1746140
7: -9.5698729, 8.9370356, -9.5695219, 8.9339333, -12.4780502, 12.5216980
8: -26.3628540, -5.5974197, -26.3547363, -5.5937438, -9.8429565, 9.8524017
9: -14.5974836, 2.1947536, -14.6003141, 2.1472003, -12.9212799, 12.9929504
10: -5.9037876, 11.7570057, -5.9089441, 11.7289772, -13.2295647, 13.2704926
11: 9.5702133, 21.1181164, 9.5660515, 21.1191463, -7.5076923, 7.5084000
12: -15.1765890, 9.8726883, -15.1721153, 9.8761921, -18.5516281, 18.4793472
13: -28.0345421, -3.0054464, -28.0470104, -3.0726566, -12.8668976, 12.9699440
14: -31.4090328, 0.6711528, -31.3892212, 0.6696637, -21.5461121, 21.5308533
15: -24.9360733, -10.6829510, -24.9404564, -10.6881304, -8.8981400, 8.9091187
16: -6.9507165, 7.9098811, -6.9506598, 7.8917942, -10.1175537, 10.1482964
17: -14.7950630, 11.7778692, -14.7653170, 11.7792301, -21.7801437, 21.7407913
18: -0.9353757, 12.5831528, -0.8854392, 12.5870457, -10.7484856, 10.6783066
19: -5.3051577, 4.7442894, -5.2886324, 4.7448864, -7.5895271, 7.5726700
20: -3.4108980, 7.9740343, -3.4075880, 7.9825659, -10.2641258, 10.2824059
21: -1.9534781, 8.8813705, -1.9446721, 8.8851652, -8.9277382, 8.9347343
22: -9.2307091, 2.7860808, -9.2122440, 2.7862687, -8.7049217, 8.6850586
23: 1.3534564, 12.5121489, 1.3496174, 12.5118895, -7.7355671, 7.7374992
24: -2.6860359, 10.5014305, -2.6870494, 10.4996624, -8.1668396, 8.1714973
25: 0.3757482, 13.7667370, 0.3373559, 13.7652798, -9.3717766, 9.4190521
26: -17.4360561, 2.4872656, -17.3992443, 2.4927273, -14.6234894, 14.5773544
27: -10.3267260, 6.2935686, -10.2687006, 6.3006601, -9.1772575, 9.1088963
28: 1.0734587, 13.5818386, 1.0847495, 13.5820456, -9.6448860, 9.6354904
29: -5.1235781, 8.3609867, -5.0919466, 8.3625441, -8.6479378, 8.6092834
30: 5.9819746, 17.7194386, 5.9695368, 17.7195358, -7.6598186, 7.6848984
31: -3.3923211, 10.3981915, -3.3825128, 10.3999825, -9.1742897, 9.1709099
32: -19.6005344, -2.7408423, -19.5951118, -2.7370543, -10.7354431, 10.7191639
33: -47.0450363, -21.5097885, -47.0490913, -21.5647717, -14.5429459, 14.6180153
34: -29.7325764, -10.5695105, -29.7293396, -10.5872917, -10.6971588, 10.6812935
35: -29.2327404, -9.9375515, -29.2325535, -9.9602985, -10.7297325, 10.7123108
36: -31.8991928, -9.3960485, -31.8891029, -9.4108906, -12.7356911, 12.7068176
37: -46.1434784, -23.4747753, -46.1474915, -23.4880638, -16.1067123, 16.0857162
38: -34.3112793, -11.5098429, -34.3071175, -11.5168276, -15.1521606, 15.1235886
39: -56.3268700, -30.6284828, -56.3321342, -30.6833038, -13.1779022, 13.2559052
40: -40.2576981, -23.2915497, -40.2581558, -23.2960052, -8.2050285, 8.2115078
41: -26.7527485, -6.9857416, -26.7452316, -6.9705858, -11.2953568, 11.2552185
42: -14.5428333, -2.0952988, -14.5427990, -2.1193805, -8.5218143, 8.5408249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0925407, upper bound: 5.0884651
time: 22.97 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0825283, upper bound: 5.0925407
time: 26.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 51.30 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 51.30
Output dim: 11, lower bound: -5.0825284, upper bound: 5.0883658
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 51.30
Output dim: 11, lower bound: -5.0825284, upper bound: 5.0924408
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 51.30
Output dim: 11, lower bound: -5.0925407, upper bound: 5.0884651
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 51.30
Output dim: 11, lower bound: -5.0825283, upper bound: 5.0925407

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.2252903, 0.7170136, -20.2234421, 0.6990862, -13.4150200, 13.4386330
1: -6.4282598, 5.2893586, -6.4229298, 5.2835417, -6.3912964, 6.4016209
2: -11.0206680, 2.2357619, -11.0136929, 2.2235641, -8.5490952, 8.5766830
3: -12.3393898, 3.4285364, -12.3383942, 3.3936014, -11.1368561, 11.1825066
4: -22.0451317, -5.6337528, -22.0420094, -5.6437149, -9.2328415, 9.2629662
5: -10.7975788, 5.6628423, -10.7931852, 5.6409879, -12.1568146, 12.1855965
6: -22.4833584, -4.4972076, -22.4650803, -4.5143762, -11.1489105, 11.1450539
7: -9.5644436, 8.9344997, -9.5616541, 8.9248638, -12.4636688, 12.5103874
8: -26.3555679, -5.5984364, -26.3441048, -5.6000381, -9.8246498, 9.8337669
9: -14.5927019, 2.1908820, -14.5911293, 2.1368809, -12.8991318, 12.9775200
10: -5.8993196, 11.7531147, -5.8974576, 11.7156792, -13.2109070, 13.2538109
11: 9.5729380, 21.1074753, 9.6011868, 21.1002312, -7.4885654, 7.4555836
12: -15.1726913, 9.8591185, -15.1603842, 9.8598471, -18.5260620, 18.4480209
13: -28.0323219, -3.0113523, -28.0316696, -3.0868213, -12.8504791, 12.9472466
14: -31.4017601, 0.6656437, -31.3526497, 0.6582749, -21.5232544, 21.4692154
15: -24.9233322, -10.6848345, -24.9277267, -10.6968708, -8.8724976, 8.8935432
16: -6.9486885, 7.9086237, -6.9462161, 7.8841925, -10.1005325, 10.1290016
17: -14.7893190, 11.7623949, -14.7369804, 11.7634382, -21.7556458, 21.6844711
18: -0.9318981, 12.5773315, -0.8718119, 12.5772552, -10.7358360, 10.6494026
19: -5.3029761, 4.7433133, -5.2723484, 4.7415528, -7.5796509, 7.5506344
20: -3.4082150, 7.9729261, -3.3954189, 7.9686704, -10.2406921, 10.2672653
21: -1.9494605, 8.8795738, -1.9266791, 8.8780994, -8.9123993, 8.9150391
22: -9.2282381, 2.7842417, -9.2030964, 2.7819581, -8.6953506, 8.6721478
23: 1.3553270, 12.5058994, 1.3831007, 12.4995928, -7.7240410, 7.6946602
24: -2.6841791, 10.4930449, -2.6475816, 10.4828882, -8.1500435, 8.1140671
25: 0.3784976, 13.7525091, 0.3990448, 13.7376757, -9.3480263, 9.3391762
26: -17.4321651, 2.4837861, -17.3810081, 2.4841654, -14.6116257, 14.5481033
27: -10.3245106, 6.2924252, -10.2633266, 6.2919860, -9.1612778, 9.0990925
28: 1.0760117, 13.5770683, 1.1148112, 13.5725622, -9.6341515, 9.5984764
29: -5.1208520, 8.3578587, -5.0817242, 8.3561964, -8.6384563, 8.5894585
30: 5.9842405, 17.7076073, 6.0022497, 17.7012463, -7.6417217, 7.6385841
31: -3.3872266, 10.3943691, -3.3563368, 10.3907804, -9.1584549, 9.1396561
32: -19.5879059, -2.7428372, -19.5696526, -2.7767437, -10.6711349, 10.6930275
33: -47.0397720, -21.5135269, -47.0379219, -21.5773392, -14.5099525, 14.5961456
34: -29.7263603, -10.5711327, -29.7186146, -10.5958862, -10.6623917, 10.6641388
35: -29.2239532, -9.9394331, -29.2215347, -9.9679098, -10.7095985, 10.6990662
36: -31.8968582, -9.3976564, -31.8841133, -9.4224129, -12.7099762, 12.6916199
37: -46.1403351, -23.4773293, -46.1352310, -23.4928093, -16.0941467, 16.0465851
38: -34.3063126, -11.5110836, -34.2964554, -11.5238991, -15.1375427, 15.1054802
39: -56.3220749, -30.6305275, -56.3219833, -30.6887817, -13.1541176, 13.2386322
40: -40.2508469, -23.2927837, -40.2447281, -23.3098106, -8.1819077, 8.1898346
41: -26.7426872, -6.9879389, -26.7258263, -7.0038843, -11.2484627, 11.2293358
42: -14.5403929, -2.0971661, -14.5336218, -2.1253905, -8.5109062, 8.5186157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0732018, upper bound: 5.0919859
time: 15.56 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0820561, upper bound: 5.0919859
time: 18.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.2165756, 0.6998730, -20.2273235, 0.7038646, -13.4276505, 13.4470139
1: -6.4258847, 5.2821374, -6.4288650, 5.2883778, -6.4010448, 6.4150257
2: -11.0173416, 2.2233195, -11.0200548, 2.2447336, -8.5706253, 8.5685177
3: -12.3275795, 3.4111652, -12.3395958, 3.4077122, -11.1468277, 11.1663551
4: -22.0363464, -5.6477838, -22.0444298, -5.6252165, -9.2478065, 9.2478828
5: -10.7941093, 5.6487308, -10.8009825, 5.6671066, -12.1788177, 12.1759033
6: -22.4874153, -4.4982362, -22.4904919, -4.4689350, -11.1881027, 11.1605186
7: -9.5554714, 8.9231834, -9.5622072, 8.9325228, -12.4617615, 12.5003052
8: -26.3430042, -5.6110334, -26.3434906, -5.5943255, -9.8223228, 9.8270874
9: -14.5813990, 2.1753414, -14.5918713, 2.1437500, -12.9024277, 12.9656258
10: -5.8953900, 11.7403269, -5.9057713, 11.7254333, -13.2179565, 13.2478981
11: 9.5874624, 21.1094284, 9.5679655, 21.1141739, -7.4869232, 7.5012894
12: -15.1465473, 9.8363686, -15.1696377, 9.8554344, -18.5000687, 18.4395294
13: -28.0275249, -3.0135801, -28.0456352, -3.0764296, -12.8570862, 12.9553146
14: -31.3769569, 0.6612282, -31.3824406, 0.6644483, -21.5077972, 21.5157013
15: -24.9266224, -10.6859703, -24.9353447, -10.6896849, -8.8816719, 8.8826084
16: -6.9457321, 7.8977776, -6.9486847, 7.8903284, -10.1087418, 10.1322689
17: -14.7635937, 11.7414875, -14.7615108, 11.7582569, -21.7283020, 21.6994553
18: -0.9125988, 12.5688248, -0.8817353, 12.5792217, -10.7152634, 10.6595383
19: -5.2981215, 4.7393336, -5.2866716, 4.7446017, -7.5794010, 7.5623627
20: -3.3995323, 7.9655147, -3.4039571, 7.9816298, -10.2505226, 10.2671471
21: -1.9435099, 8.8754311, -1.9413860, 8.8844566, -8.9140053, 8.9204254
22: -9.2210693, 2.7852769, -9.2105989, 2.7851195, -8.6935463, 8.6752396
23: 1.3633773, 12.5095081, 1.3514125, 12.5117397, -7.7248650, 7.7315350
24: -2.6781247, 10.4984360, -2.6854594, 10.4987822, -8.1549854, 8.1619625
25: 0.3876483, 13.7629395, 0.3395858, 13.7645473, -9.3580551, 9.4107742
26: -17.4118366, 2.4751482, -17.3961487, 2.4859221, -14.5903893, 14.5612488
27: -10.3163481, 6.2914772, -10.2653742, 6.3003020, -9.1616783, 9.1024952
28: 1.0852842, 13.5783224, 1.0870655, 13.5818214, -9.6318016, 9.6283455
29: -5.1084299, 8.3535748, -5.0901117, 8.3591061, -8.6298027, 8.5995827
30: 5.9979172, 17.7138672, 5.9710484, 17.7166100, -7.6488953, 7.6762791
31: -3.3811665, 10.3909245, -3.3779216, 10.3997211, -9.1600533, 9.1546707
32: -19.5929871, -2.7498958, -19.5930500, -2.7389629, -10.7193909, 10.7052231
33: -47.0290337, -21.5366516, -47.0407143, -21.5681534, -14.5269508, 14.5807686
34: -29.7160683, -10.5829353, -29.7208614, -10.5886374, -10.6811371, 10.6546249
35: -29.2243919, -9.9489727, -29.2288723, -9.9619236, -10.7188683, 10.7004967
36: -31.8886528, -9.3975945, -31.8866158, -9.4121094, -12.7177658, 12.7008247
37: -46.1329193, -23.4856567, -46.1449661, -23.4930058, -16.0732727, 16.0661774
38: -34.2980080, -11.5124931, -34.3032875, -11.5177660, -15.1250305, 15.1136208
39: -56.3093452, -30.6463661, -56.3225212, -30.6852875, -13.1651649, 13.2260628
40: -40.2526932, -23.2982216, -40.2558250, -23.2969856, -8.1843910, 8.1997681
41: -26.7458973, -6.9933624, -26.7434654, -6.9722090, -11.2744598, 11.2430992
42: -14.5414085, -2.1017168, -14.5413723, -2.1211076, -8.4969444, 8.5276222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0831649, upper bound: 5.0879579
time: 21.24 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0920332, upper bound: 5.0879579
time: 21.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2257767, 0.7149758, -20.2319221, 0.7043607, -13.4299049, 13.4661446
1: -6.4284663, 5.2880955, -6.4299712, 5.2885504, -6.4016953, 6.4219990
2: -11.0250750, 2.2360072, -11.0243587, 2.2452497, -8.5759583, 8.5847931
3: -12.3415747, 3.4287219, -12.3472328, 3.4084466, -11.1555748, 11.1911964
4: -22.0491104, -5.6333723, -22.0513992, -5.6246543, -9.2566261, 9.2695770
5: -10.8030109, 5.6632538, -10.8057137, 5.6677823, -12.1885681, 12.1957779
6: -22.4957714, -4.4971819, -22.4915180, -4.4699368, -11.2059822, 11.1658020
7: -9.5658216, 8.9355583, -9.5671053, 8.9330215, -12.4714661, 12.5168533
8: -26.3569908, -5.5980973, -26.3511887, -5.5941620, -9.8275604, 9.8472595
9: -14.5931168, 2.1912110, -14.5976467, 2.1450632, -12.9081268, 12.9875603
10: -5.8998833, 11.7539139, -5.9066133, 11.7270870, -13.2224388, 13.2653542
11: 9.5717249, 21.1158142, 9.5669708, 21.1176338, -7.5023766, 7.4996147
12: -15.1741333, 9.8595829, -15.1705980, 9.8683043, -18.5406647, 18.4609528
13: -28.0329323, -3.0083477, -28.0460396, -3.0743656, -12.8634148, 12.9660149
14: -31.4026546, 0.6688972, -31.3853531, 0.6683538, -21.5394287, 21.5163879
15: -24.9235992, -10.6841183, -24.9328194, -10.6888275, -8.8817177, 8.9025707
16: -6.9493523, 7.9079366, -6.9498448, 7.8906479, -10.1094818, 10.1439590
17: -14.7900782, 11.7655621, -14.7623005, 11.7711754, -21.7672882, 21.7169266
18: -0.9322982, 12.5780373, -0.8835943, 12.5838928, -10.7437172, 10.6643791
19: -5.3035984, 4.7440486, -5.2876654, 4.7447457, -7.5841408, 7.5697460
20: -3.4081783, 7.9731770, -3.4059448, 7.9820495, -10.2592506, 10.2797470
21: -1.9505016, 8.8805437, -1.9428339, 8.8846645, -8.9204865, 8.9311790
22: -9.2287874, 2.7852893, -9.2110767, 2.7857871, -8.7021084, 8.6819019
23: 1.3548518, 12.5119905, 1.3504754, 12.5118027, -7.7332306, 7.7344303
24: -2.6845427, 10.5006142, -2.6861253, 10.4991741, -8.1620998, 8.1652393
25: 0.3779216, 13.7661676, 0.3386800, 13.7649097, -9.3684349, 9.4146118
26: -17.4330807, 2.4840627, -17.3973694, 2.4907818, -14.6189880, 14.5673904
27: -10.3240633, 6.2931585, -10.2670670, 6.3003821, -9.1739826, 9.1056938
28: 1.0753367, 13.5815973, 1.0858963, 13.5819206, -9.6419945, 9.6329269
29: -5.1213455, 8.3590345, -5.0906000, 8.3613873, -8.6441059, 8.6004601
30: 5.9833989, 17.7145844, 5.9703941, 17.7165184, -7.6539803, 7.6788311
31: -3.3880999, 10.3979931, -3.3799644, 10.3998623, -9.1671028, 9.1670437
32: -19.5989037, -2.7425280, -19.5941277, -2.7380867, -10.7267418, 10.7133141
33: -47.0416374, -21.5127525, -47.0469627, -21.5665894, -14.5226936, 14.6078682
34: -29.7294922, -10.5706778, -29.7273483, -10.5880537, -10.6781387, 10.6719360
35: -29.2256737, -9.9389820, -29.2282619, -9.9612007, -10.7225571, 10.7058372
36: -31.8966827, -9.3971310, -31.8875771, -9.4115524, -12.7322884, 12.7005920
37: -46.1415634, -23.4787750, -46.1462860, -23.4905071, -16.1038361, 16.0714722
38: -34.3081436, -11.5108433, -34.3051682, -11.5174084, -15.1485901, 15.1150208
39: -56.3233643, -30.6303101, -56.3299484, -30.6844177, -13.1607437, 13.2466850
40: -40.2561646, -23.2926846, -40.2571907, -23.2966938, -8.2002640, 8.1993904
41: -26.7514153, -6.9875364, -26.7443790, -6.9716396, -11.2896194, 11.2450562
42: -14.5417910, -2.0969281, -14.5421810, -2.1203778, -8.5175800, 8.5277996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0831649, upper bound: 5.0920333
time: 11.48 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0920332, upper bound: 5.0920333
time: 19.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 33.36 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.36
Output dim: 11, lower bound: -5.0732018, upper bound: 5.0919859
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.36
Output dim: 11, lower bound: -5.0820561, upper bound: 5.0919859
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.36
Output dim: 11, lower bound: -5.0831649, upper bound: 5.0879579
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.36
Output dim: 11, lower bound: -5.0920332, upper bound: 5.0879579
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.36
Output dim: 11, lower bound: -5.0831649, upper bound: 5.0920333
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.36
Output dim: 11, lower bound: -5.0920332, upper bound: 5.0920333

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -20.2244263, 0.7071009, -20.2220383, 0.6829755, -13.3964043, 13.4265862
1: -6.4280729, 5.2791758, -6.4226327, 5.2664509, -6.3737602, 6.3909531
2: -11.0149536, 2.2339847, -11.0043049, 2.2205997, -8.5399170, 8.5651093
3: -12.3369770, 3.4279797, -12.3346453, 3.3927147, -11.1332016, 11.1781693
4: -22.0440884, -5.6362805, -22.0402756, -5.6478705, -9.2248650, 9.2567863
5: -10.7921343, 5.6618419, -10.7842865, 5.6392775, -12.1484909, 12.1747169
6: -22.4595795, -4.4977503, -22.4252014, -4.5152063, -11.1238556, 11.1049881
7: -9.5640697, 8.9310474, -9.5610590, 8.9191113, -12.4572601, 12.5060158
8: -26.3549957, -5.6082659, -26.3431873, -5.6164598, -9.8072586, 9.8228035
9: -14.5914230, 2.1823106, -14.5890360, 2.1225181, -12.8819809, 12.9658585
10: -5.8984051, 11.7400112, -5.8959007, 11.6939526, -13.1863785, 13.2377052
11: 9.5738573, 21.0974846, 9.6026545, 21.0834999, -7.4710846, 7.4439373
12: -15.1655693, 9.8582125, -15.1483746, 9.8583946, -18.5174942, 18.4355927
13: -28.0269489, -3.0135713, -28.0228310, -3.0904357, -12.8387566, 12.9359398
14: -31.4003487, 0.6526198, -31.3503189, 0.6369123, -21.4983597, 21.4524994
15: -24.9225941, -10.6929836, -24.9265232, -10.7103672, -8.8575935, 8.8836975
16: -6.9478436, 7.9038916, -6.9448290, 7.8764062, -10.0914154, 10.1225147
17: -14.7878952, 11.7602673, -14.7345743, 11.7599506, -21.7497787, 21.6788330
18: -0.9303162, 12.5759354, -0.8691697, 12.5750704, -10.7308426, 10.6432762
19: -5.3014312, 4.7427206, -5.2698975, 4.7405329, -7.5764370, 7.5467606
20: -3.4040956, 7.9718428, -3.3885694, 7.9668627, -10.2338448, 10.2586174
21: -1.9480944, 8.8769627, -1.9244406, 8.8738823, -8.9069557, 8.9101791
22: -9.2253351, 2.7832775, -9.1983337, 2.7804182, -8.6888905, 8.6645565
23: 1.3563980, 12.5011148, 1.3848101, 12.4915657, -7.7149391, 7.6880722
24: -2.6833403, 10.4796238, -2.6461928, 10.4604759, -8.1269302, 8.0993385
25: 0.3791552, 13.7358189, 0.4000878, 13.7097940, -9.3198814, 9.3217049
26: -17.4298477, 2.4831755, -17.3772297, 2.4831257, -14.6084633, 14.5437546
27: -10.3192558, 6.2911472, -10.2546949, 6.2897329, -9.1515865, 9.0842247
28: 1.0785615, 13.5736103, 1.1189663, 13.5667772, -9.6251450, 9.5900497
29: -5.1178432, 8.3545141, -5.0768127, 8.3505917, -8.6266308, 8.5775757
30: 5.9847851, 17.6930828, 6.0031214, 17.6770058, -7.6168556, 7.6230392
31: -3.3862340, 10.3927546, -3.3547063, 10.3880606, -9.1541595, 9.1358147
32: -19.5622139, -2.7433434, -19.5268440, -2.7776062, -10.6438942, 10.6490059
33: -47.0325165, -21.5154362, -47.0256767, -21.5804443, -14.4972763, 14.5803757
34: -29.7155685, -10.5715103, -29.7007084, -10.5964756, -10.6507225, 10.6454430
35: -29.2114811, -9.9398565, -29.2006474, -9.9686499, -10.6959152, 10.6776543
36: -31.8718147, -9.3981228, -31.8422909, -9.4230576, -12.6837463, 12.6490898
37: -46.1384888, -23.4780483, -46.1321106, -23.4939232, -16.0888748, 16.0410995
38: -34.2860718, -11.5112915, -34.2627602, -11.5243177, -15.1165695, 15.0716438
39: -56.3124008, -30.6316223, -56.3057709, -30.6905746, -13.1369896, 13.2148705
40: -40.2450104, -23.2929058, -40.2349625, -23.3100185, -8.1758919, 8.1802368
41: -26.7224503, -6.9883671, -26.6920700, -7.0046263, -11.2268085, 11.1944504
42: -14.5371599, -2.0978186, -14.5283146, -2.1264439, -8.5058651, 8.5115623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1465

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0727880, upper bound: 5.0863520
time: 15.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0727880, upper bound: 5.0915594
time: 19.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -20.2248230, 0.7163460, -20.2419987, 0.6999674, -13.4076157, 13.4584236
1: -6.4281936, 5.2887964, -6.4427319, 5.2841892, -6.3861523, 6.4207096
2: -11.0201206, 2.2356122, -11.0138426, 2.2313907, -8.5542679, 8.5735626
3: -12.3381023, 3.4283996, -12.3396511, 3.3970726, -11.1373024, 11.1836433
4: -22.0449600, -5.6342974, -22.0468483, -5.6431961, -9.2288513, 9.2694206
5: -10.7968683, 5.6626649, -10.7937498, 5.6541014, -12.1676025, 12.1823730
6: -22.4819717, -4.4973412, -22.4645653, -4.4738750, -11.1872444, 11.1325150
7: -9.5639582, 8.9342184, -9.5669603, 8.9263687, -12.4633636, 12.5154343
8: -26.3554287, -5.5989561, -26.3646793, -5.5997577, -9.8189926, 9.8534966
9: -14.5925398, 2.1883616, -14.6050148, 2.1355741, -12.8938675, 12.9945755
10: -5.8991733, 11.7503147, -5.9167643, 11.7148190, -13.2033615, 13.2759628
11: 9.5732841, 21.1068897, 9.5839167, 21.1001854, -7.4823046, 7.4729118
12: -15.1722202, 9.8582802, -15.1629467, 9.8759537, -18.5419846, 18.4462891
13: -28.0311527, -3.0118210, -28.0310555, -3.0753183, -12.8522758, 12.9462700
14: -31.4011345, 0.6647339, -31.3792419, 0.6589646, -21.5157089, 21.4968414
15: -24.9231987, -10.6871338, -24.9412842, -10.6990862, -8.8666954, 8.9105473
16: -6.9483128, 7.9080234, -6.9559140, 7.8849087, -10.0970345, 10.1385994
17: -14.7890577, 11.7611666, -14.7431841, 11.7627449, -21.7560654, 21.6870728
18: -0.9312530, 12.5770893, -0.8768973, 12.5797758, -10.7412910, 10.6532707
19: -5.3025475, 4.7432537, -5.2745571, 4.7409239, -7.5784874, 7.5524063
20: -3.4069633, 7.9727607, -3.3960600, 7.9770789, -10.2476234, 10.2649879
21: -1.9491148, 8.8793221, -1.9326644, 8.8784676, -8.9111671, 8.9191113
22: -9.2278728, 2.7839460, -9.2053957, 2.7821684, -8.6938324, 8.6724281
23: 1.3557123, 12.5056353, 1.3713126, 12.4992971, -7.7188854, 7.7062378
24: -2.6839104, 10.4922676, -2.6721511, 10.4827061, -8.1441669, 8.1385918
25: 0.3787427, 13.7515783, 0.3673375, 13.7368383, -9.3391457, 9.3700447
26: -17.4316769, 2.4821043, -17.3817139, 2.4832199, -14.6108551, 14.5489655
27: -10.3213215, 6.2922244, -10.2623081, 6.2917233, -9.1658421, 9.0998058
28: 1.0765703, 13.5768566, 1.1097164, 13.5723705, -9.6301880, 9.6033974
29: -5.1206102, 8.3566160, -5.0842171, 8.3552179, -8.6352501, 8.5879364
30: 5.9845028, 17.7067871, 5.9755893, 17.7009583, -7.6326180, 7.6645794
31: -3.3862848, 10.3941975, -3.3574924, 10.3875341, -9.1561966, 9.1412582
32: -19.5864563, -2.7430050, -19.5701942, -2.7342186, -10.7117538, 10.6805115
33: -47.0393066, -21.5138702, -47.0382996, -21.5630360, -14.5199928, 14.5892487
34: -29.7256088, -10.5713215, -29.7204170, -10.5839062, -10.6736259, 10.6616669
35: -29.2231655, -9.9394932, -29.2219276, -9.9478426, -10.7299843, 10.6918182
36: -31.8954544, -9.3978281, -31.8842087, -9.3855219, -12.7455521, 12.6776810
37: -46.1399307, -23.4796753, -46.1314545, -23.4931755, -16.0939865, 16.0440598
38: -34.3050499, -11.5112238, -34.2981529, -11.4901686, -15.1706390, 15.0941849
39: -56.3213501, -30.6309052, -56.3222580, -30.6723347, -13.1647110, 13.2241478
40: -40.2503815, -23.2928619, -40.2451057, -23.2995052, -8.1914368, 8.1876335
41: -26.7414436, -6.9881763, -26.7262287, -6.9758062, -11.2750702, 11.2209930
42: -14.5399771, -2.0974889, -14.5346470, -2.1199188, -8.5162983, 8.5170403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1465

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0727880, upper bound: 5.0863520
time: 29.23 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0816383, upper bound: 5.0915594
time: 17.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.2161808, 0.6991909, -20.2458763, 0.7047169, -13.4202843, 13.4668503
1: -6.4258204, 5.2815371, -6.4486542, 5.2891064, -6.3959084, 6.4341125
2: -11.0167971, 2.2231877, -11.0201283, 2.2525973, -8.5758095, 8.5653572
3: -12.3262920, 3.4110420, -12.3407974, 3.4111745, -11.1473007, 11.1674118
4: -22.0361767, -5.6483164, -22.0492420, -5.6247301, -9.2438164, 9.2543297
5: -10.7934141, 5.6485372, -10.8015518, 5.6802893, -12.1896210, 12.1726646
6: -22.4860249, -4.4983778, -22.4900513, -4.4283848, -11.2265244, 11.1480637
7: -9.5549479, 8.9228992, -9.5675306, 8.9341621, -12.4614563, 12.5053291
8: -26.3428650, -5.6115689, -26.3640652, -5.5940185, -9.8166656, 9.8468246
9: -14.5812454, 2.1728106, -14.6057215, 2.1424201, -12.8971481, 12.9826775
10: -5.8952298, 11.7375221, -5.9250298, 11.7245636, -13.2104416, 13.2700615
11: 9.5877819, 21.1088390, 9.5507011, 21.1140862, -7.4806061, 7.5185966
12: -15.1460543, 9.8355274, -15.1722584, 9.8715954, -18.5160217, 18.4378281
13: -28.0263634, -3.0140326, -28.0451221, -3.0649033, -12.8589172, 12.9543571
14: -31.3763866, 0.6602929, -31.4091110, 0.6651421, -21.5002518, 21.5434418
15: -24.9264946, -10.6882734, -24.9488640, -10.6918793, -8.8758774, 8.8996658
16: -6.9453621, 7.8971920, -6.9584036, 7.8910518, -10.1052475, 10.1418819
17: -14.7632885, 11.7402601, -14.7677860, 11.7574940, -21.7286835, 21.7021332
18: -0.9119203, 12.5685930, -0.8868887, 12.5816803, -10.7206802, 10.6634521
19: -5.2977142, 4.7392597, -5.2889066, 4.7439961, -7.5782433, 7.5641537
20: -3.3983021, 7.9653435, -3.4045610, 7.9900589, -10.2574692, 10.2648888
21: -1.9431508, 8.8751793, -1.9473985, 8.8848572, -8.9127808, 8.9244957
22: -9.2207031, 2.7849970, -9.2129345, 2.7853227, -8.6920433, 8.6755428
23: 1.3637766, 12.5092449, 1.3395932, 12.5114412, -7.7196903, 7.7431488
24: -2.6778657, 10.4976664, -2.7100890, 10.4985943, -8.1490726, 8.1865387
25: 0.3878460, 13.7620087, 0.3078876, 13.7636909, -9.3491726, 9.4416733
26: -17.4113884, 2.4734550, -17.3967838, 2.4849339, -14.5896225, 14.5621872
27: -10.3131409, 6.2912793, -10.2643890, 6.3000598, -9.1662712, 9.1032333
28: 1.0858381, 13.5780993, 1.0819561, 13.5816154, -9.6278534, 9.6333008
29: -5.1081710, 8.3523321, -5.0925970, 8.3580627, -8.6266212, 8.5980835
30: 5.9981356, 17.7130394, 5.9443789, 17.7162628, -7.6397667, 7.7022552
31: -3.3802383, 10.3907423, -3.3791091, 10.3964739, -9.1578026, 9.1563187
32: -19.5915260, -2.7500579, -19.5935860, -2.6963732, -10.7600632, 10.6927376
33: -47.0285416, -21.5370216, -47.0410919, -21.5538483, -14.5370445, 14.5739174
34: -29.7153111, -10.5831280, -29.7226219, -10.5766048, -10.6924095, 10.6521988
35: -29.2236176, -9.9490938, -29.2293243, -9.9418697, -10.7392502, 10.6933441
36: -31.8872299, -9.3977318, -31.8867874, -9.3752499, -12.7533150, 12.6868591
37: -46.1325607, -23.4879646, -46.1411591, -23.4933987, -16.0731201, 16.0637894
38: -34.2967796, -11.5126448, -34.3049469, -11.4840097, -15.1581039, 15.1023483
39: -56.3086624, -30.6467590, -56.3228607, -30.6688213, -13.1757812, 13.2116585
40: -40.2522049, -23.2982998, -40.2562065, -23.2866459, -8.1939240, 8.1975746
41: -26.7446556, -6.9935722, -26.7438564, -6.9441395, -11.3011017, 11.2347870
42: -14.5409803, -2.1019964, -14.5424547, -2.1155968, -8.5023518, 8.5260620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1465

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0916037, upper bound: 5.0822871
time: 30.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0916037, upper bound: 5.0875253
time: 19.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -20.2248669, 0.7050996, -20.2304916, 0.6882761, -13.4113083, 13.4541130
1: -6.4282794, 5.2779236, -6.4296856, 5.2714891, -6.3841610, 6.4113369
2: -11.0193214, 2.2342322, -11.0150242, 2.2423284, -8.5667877, 8.5732384
3: -12.3391705, 3.4281950, -12.3434868, 3.4075549, -11.1519318, 11.1868210
4: -22.0480747, -5.6358967, -22.0496216, -5.6288118, -9.2486343, 9.2633171
5: -10.7975864, 5.6622081, -10.7968540, 5.6661196, -12.1804199, 12.1849289
6: -22.4720135, -4.4976864, -22.4516659, -4.4707093, -11.1809731, 11.1257439
7: -9.5654888, 8.9321146, -9.5664358, 8.9274406, -12.4651031, 12.5125122
8: -26.3564339, -5.6079550, -26.3501701, -5.6105156, -9.8101540, 9.8362694
9: -14.5918303, 2.1826115, -14.5954647, 2.1306908, -12.8910294, 12.9758644
10: -5.8989792, 11.7407751, -5.9049530, 11.7053585, -13.1979904, 13.2491875
11: 9.5726242, 21.1057892, 9.5684834, 21.1008549, -7.4849081, 7.4879341
12: -15.1670456, 9.8587675, -15.1586781, 9.8668480, -18.5320892, 18.4485245
13: -28.0275955, -3.0105188, -28.0372562, -3.0779696, -12.8516693, 12.9547234
14: -31.4012585, 0.6558287, -31.3831024, 0.6469674, -21.5144730, 21.4998016
15: -24.9228687, -10.6922550, -24.9315376, -10.7023220, -8.8667984, 8.8928108
16: -6.9485006, 7.9032259, -6.9484057, 7.8828802, -10.1003723, 10.1374760
17: -14.7886629, 11.7633896, -14.7600307, 11.7676821, -21.7614517, 21.7113953
18: -0.9307182, 12.5766706, -0.8810501, 12.5815735, -10.7386246, 10.6583519
19: -5.3020763, 4.7434573, -5.2852054, 4.7437472, -7.5808983, 7.5658951
20: -3.4040489, 7.9720969, -3.3991008, 7.9803576, -10.2524605, 10.2710876
21: -1.9491423, 8.8779087, -1.9406099, 8.8803673, -8.9151688, 8.9262581
22: -9.2259083, 2.7843513, -9.2063560, 2.7842360, -8.6956902, 8.6743584
23: 1.3559117, 12.5071754, 1.3522000, 12.5037451, -7.7240620, 7.7278557
24: -2.6837082, 10.4871559, -2.6847479, 10.4767437, -8.1389732, 8.1505032
25: 0.3785686, 13.7494783, 0.3397501, 13.7369337, -9.3402691, 9.3971481
26: -17.4308167, 2.4834344, -17.3935986, 2.4897370, -14.6158104, 14.5630493
27: -10.3188038, 6.2918139, -10.2584324, 6.2981672, -9.1643677, 9.0908432
28: 1.0778825, 13.5781384, 1.0900238, 13.5761042, -9.6329422, 9.6245155
29: -5.1183338, 8.3556576, -5.0857258, 8.3557415, -8.6322918, 8.5886269
30: 5.9839225, 17.7000351, 5.9712858, 17.6922169, -7.6291428, 7.6632614
31: -3.3870931, 10.3963909, -3.3783429, 10.3971176, -9.1627769, 9.1632042
32: -19.5732536, -2.7430792, -19.5512733, -2.7389002, -10.6995239, 10.6693001
33: -47.0343285, -21.5146618, -47.0346756, -21.5697460, -14.5100517, 14.5921211
34: -29.7187042, -10.5710802, -29.7092896, -10.5886230, -10.6664734, 10.6531715
35: -29.2132721, -9.9394531, -29.2076111, -9.9619503, -10.7088661, 10.6846962
36: -31.8717556, -9.3975220, -31.8457737, -9.4122248, -12.7059822, 12.6580353
37: -46.1396866, -23.4794693, -46.1435013, -23.4916039, -16.0985565, 16.0664749
38: -34.2878647, -11.5110960, -34.2714729, -11.5178337, -15.1276016, 15.0813217
39: -56.3137207, -30.6314659, -56.3140373, -30.6862144, -13.1436539, 13.2229271
40: -40.2503128, -23.2927933, -40.2474442, -23.2968845, -8.1942558, 8.1897659
41: -26.7311325, -6.9879589, -26.7106342, -6.9723597, -11.2679367, 11.2102051
42: -14.5385723, -2.0975428, -14.5368547, -2.1213899, -8.5125561, 8.5207710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1465

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0827450, upper bound: 5.0863948
time: 21.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0827450, upper bound: 5.0916037
time: 20.23 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20.2253113, 0.7143185, -20.2504978, 0.7052636, -13.4225388, 13.4859734
1: -6.4284086, 5.2875137, -6.4497719, 5.2892566, -6.3965721, 6.4410973
2: -11.0245409, 2.2358506, -11.0244751, 2.2531137, -8.5811577, 8.5816193
3: -12.3403206, 3.4285784, -12.3483963, 3.4119377, -11.1560707, 11.1922722
4: -22.0489540, -5.6339025, -22.0562153, -5.6241169, -9.2525978, 9.2760162
5: -10.8023272, 5.6630163, -10.8063145, 5.6809340, -12.1994095, 12.1925430
6: -22.4943848, -4.4973097, -22.4911098, -4.4293518, -11.2443886, 11.1533279
7: -9.5653124, 8.9353008, -9.5723658, 8.9346056, -12.4711914, 12.5218925
8: -26.3568344, -5.5986509, -26.3717499, -5.5938234, -9.8219109, 9.8670006
9: -14.5929623, 2.1886952, -14.6115265, 2.1437457, -12.9028854, 13.0045815
10: -5.8997574, 11.7511072, -5.9258728, 11.7262192, -13.2149315, 13.2874985
11: 9.5720654, 21.1152191, 9.5497208, 21.1175346, -7.4960670, 7.5169392
12: -15.1736584, 9.8587589, -15.1732655, 9.8844318, -18.5566330, 18.4592056
13: -28.0317345, -3.0088036, -28.0454960, -3.0628562, -12.8652840, 12.9650612
14: -31.4020615, 0.6680088, -31.4120235, 0.6690164, -21.5318680, 21.5441284
15: -24.9234734, -10.6864042, -24.9463463, -10.6910305, -8.8759232, 8.9196320
16: -6.9489679, 7.9073534, -6.9595513, 7.8913474, -10.1060066, 10.1535835
17: -14.7897940, 11.7643318, -14.7685461, 11.7705250, -21.7676773, 21.7196732
18: -0.9316294, 12.5778284, -0.8887482, 12.5863628, -10.7491417, 10.6682892
19: -5.3032055, 4.7439780, -5.2899017, 4.7441373, -7.5829830, 7.5715313
20: -3.4069343, 7.9730091, -3.4065497, 7.9905152, -10.2662086, 10.2774849
21: -1.9501482, 8.8803005, -1.9488630, 8.8850288, -8.9192810, 8.9352474
22: -9.2284260, 2.7850204, -9.2133932, 2.7859888, -8.7005997, 8.6821995
23: 1.3552382, 12.5116968, 1.3386698, 12.5114899, -7.7280502, 7.7460442
24: -2.6842945, 10.4998369, -2.7107646, 10.4989758, -8.1561985, 8.1898022
25: 0.3781202, 13.7652130, 0.3069646, 13.7640257, -9.3595581, 9.4455109
26: -17.4325771, 2.4824038, -17.3980370, 2.4898252, -14.6182098, 14.5683212
27: -10.3208714, 6.2929282, -10.2660561, 6.3001709, -9.1785755, 9.1064320
28: 1.0758798, 13.5813885, 1.0807514, 13.5816841, -9.6380348, 9.6378670
29: -5.1211109, 8.3577633, -5.0930634, 8.3603325, -8.6409035, 8.5989456
30: 5.9836535, 17.7137566, 5.9437418, 17.7161713, -7.6448574, 7.7048168
31: -3.3871665, 10.3978195, -3.3811588, 10.3966160, -9.1648483, 9.1686535
32: -19.5974464, -2.7426770, -19.5946693, -2.6954920, -10.7674217, 10.7008209
33: -47.0411415, -21.5131741, -47.0473366, -21.5522842, -14.5328140, 14.6010094
34: -29.7287750, -10.5709209, -29.7290916, -10.5760422, -10.6894264, 10.6695023
35: -29.2249050, -9.9391365, -29.2287827, -9.9411545, -10.7429314, 10.6987000
36: -31.8953514, -9.3972569, -31.8877316, -9.3746452, -12.7678375, 12.6866074
37: -46.1411743, -23.4811039, -46.1425781, -23.4908524, -16.1037140, 16.0691147
38: -34.3068924, -11.5109520, -34.3068428, -11.4836788, -15.1816864, 15.1037865
39: -56.3226814, -30.6307907, -56.3302765, -30.6679668, -13.1713829, 13.2322807
40: -40.2556725, -23.2927818, -40.2575912, -23.2863770, -8.2097816, 8.1972084
41: -26.7501450, -6.9877219, -26.7447815, -6.9435520, -11.3162518, 11.2367592
42: -14.5413761, -2.0972223, -14.5432549, -2.1148491, -8.5230141, 8.5262508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1465

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0916037, upper bound: 5.0863947
time: 25.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0916037, upper bound: 5.0916037
time: 21.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 49.32 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0727880, upper bound: 5.0863520
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0727880, upper bound: 5.0915594
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0727880, upper bound: 5.0863520
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0816383, upper bound: 5.0915594
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0916037, upper bound: 5.0822871
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0916037, upper bound: 5.0875253
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0827450, upper bound: 5.0863948
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0827450, upper bound: 5.0916037
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0916037, upper bound: 5.0863947
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 49.32
Output dim: 11, lower bound: -5.0916037, upper bound: 5.0916037

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.2239475, 0.7003264, -20.2217064, 0.6789830, -13.3919907, 13.4052467
1: -6.4279318, 5.2759304, -6.4225340, 5.2645350, -6.3730030, 6.3765392
2: -11.0147762, 2.2281289, -11.0042076, 2.2171621, -8.5367050, 8.5438709
3: -12.3365736, 3.4237757, -12.3344059, 3.3902187, -11.1316833, 11.1602936
4: -22.0439091, -5.6400070, -22.0401649, -5.6500430, -9.2218628, 9.2482376
5: -10.7917309, 5.6552801, -10.7840376, 5.6354012, -12.1447601, 12.1510391
6: -22.4587784, -4.4980488, -22.4247284, -4.5153847, -11.1227608, 11.1035805
7: -9.5635605, 8.9234943, -9.5607872, 8.9146976, -12.4544220, 12.4750671
8: -26.3547592, -5.6138020, -26.3430347, -5.6196904, -9.8064804, 9.7961540
9: -14.5866814, 2.1815312, -14.5861883, 2.1220424, -12.8708191, 12.9626694
10: -5.8962741, 11.7390041, -5.8945732, 11.6933422, -13.1823425, 13.2356071
11: 9.5744925, 21.0954514, 9.6030550, 21.0823269, -7.4698000, 7.4339828
12: -15.1574850, 9.8578987, -15.1435881, 9.8581753, -18.4924316, 18.4324722
13: -28.0228119, -3.0142057, -28.0203991, -3.0908766, -12.8384705, 12.9300423
14: -31.3987999, 0.6461990, -31.3494244, 0.6331663, -21.4933548, 21.4419632
15: -24.9214478, -10.6935062, -24.9258480, -10.7106848, -8.8410110, 8.8821812
16: -6.9474392, 7.9037266, -6.9445705, 7.8763046, -10.0911751, 10.1202831
17: -14.7871456, 11.7551060, -14.7341175, 11.7569208, -21.7454376, 21.6668930
18: -0.9296076, 12.5716696, -0.8687799, 12.5725479, -10.7283287, 10.6375694
19: -5.3008509, 4.7415333, -5.2695465, 4.7398300, -7.5752468, 7.5446892
20: -3.4033673, 7.9699354, -3.3881307, 7.9657288, -10.2325935, 10.2562904
21: -1.9474500, 8.8768978, -1.9240518, 8.8738222, -8.9067764, 8.9078255
22: -9.2234898, 2.7829399, -9.1970577, 2.7801991, -8.6735878, 8.6623821
23: 1.3570039, 12.4994030, 1.3851616, 12.4905777, -7.7132645, 7.6829929
24: -2.6829982, 10.4778719, -2.6459928, 10.4594440, -8.1254234, 8.0977974
25: 0.3824327, 13.7356396, 0.4019954, 13.7096519, -9.3138237, 9.3172684
26: -17.4246368, 2.4827156, -17.3741703, 2.4828510, -14.5926895, 14.5405045
27: -10.3188000, 6.2877789, -10.2544298, 6.2877369, -9.1498661, 9.0786934
28: 1.0798101, 13.5734940, 1.1197073, 13.5667257, -9.6252594, 9.5857620
29: -5.1160932, 8.3543015, -5.0757771, 8.3504658, -8.6182537, 8.5761833
30: 5.9859748, 17.6929092, 6.0038519, 17.6768990, -7.6154175, 7.6205349
31: -3.3856170, 10.3905125, -3.3543336, 10.3867197, -9.1523285, 9.1304893
32: -19.5613060, -2.7436781, -19.5263023, -2.7777779, -10.6422005, 10.6482620
33: -47.0272751, -21.5160408, -47.0221710, -21.5808754, -14.4725266, 14.5773048
34: -29.7116928, -10.5718985, -29.6983795, -10.5966740, -10.6416664, 10.6427574
35: -29.2081051, -9.9400368, -29.1986294, -9.9687452, -10.6724854, 10.6765327
36: -31.8685455, -9.3981876, -31.8403244, -9.4231253, -12.6717873, 12.6475754
37: -46.1311646, -23.4786968, -46.1277771, -23.4943199, -16.0567245, 16.0386047
38: -34.2856293, -11.5163593, -34.2625122, -11.5272865, -15.1134949, 15.0627823
39: -56.3105965, -30.6322365, -56.3047409, -30.6909981, -13.1190224, 13.2112427
40: -40.2413292, -23.2933941, -40.2327652, -23.3103218, -8.1638069, 8.1773663
41: -26.7204971, -6.9891338, -26.6909008, -7.0050163, -11.2243690, 11.1925697
42: -14.5346317, -2.0984421, -14.5267668, -2.1268203, -8.5006409, 8.5098495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 3.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0629586, upper bound: 5.0912202
time: 24.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0724463, upper bound: 5.0912203
time: 20.08 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2243309, 0.7095089, -20.2417145, 0.6959889, -13.4031944, 13.4370651
1: -6.4280529, 5.2855606, -6.4426489, 5.2822719, -6.3853798, 6.4062843
2: -11.0199661, 2.2297223, -11.0137310, 2.2279313, -8.5510559, 8.5523415
3: -12.3377295, 3.4241674, -12.3394203, 3.3945708, -11.1357689, 11.1657257
4: -22.0447807, -5.6379981, -22.0467453, -5.6453590, -9.2258224, 9.2608795
5: -10.7964802, 5.6561432, -10.7935543, 5.6502404, -12.1638947, 12.1586990
6: -22.4811478, -4.4976702, -22.4640865, -4.4740763, -11.1861343, 11.1311111
7: -9.5634451, 8.9266682, -9.5667191, 8.9219666, -12.4605103, 12.4844398
8: -26.3551769, -5.6044607, -26.3645592, -5.6030345, -9.8181992, 9.8268204
9: -14.5877838, 2.1876106, -14.6022072, 2.1350992, -12.8826675, 12.9913940
10: -5.8970418, 11.7492990, -5.9154282, 11.7142181, -13.1993637, 13.2738838
11: 9.5739136, 21.1048832, 9.5842934, 21.0989990, -7.4810133, 7.4629612
12: -15.1641254, 9.8579350, -15.1582232, 9.8757935, -18.5169449, 18.4431839
13: -28.0269718, -3.0125136, -28.0285797, -3.0757272, -12.8520050, 12.9403839
14: -31.3996181, 0.6583920, -31.3783035, 0.6551275, -21.5107269, 21.4862823
15: -24.9220524, -10.6876631, -24.9406013, -10.6994114, -8.8501129, 8.9090195
16: -6.9479070, 7.9078541, -6.9556799, 7.8848124, -10.0967979, 10.1363525
17: -14.7882690, 11.7559414, -14.7427006, 11.7596531, -21.7517166, 21.6751404
18: -0.9305475, 12.5728321, -0.8764911, 12.5772762, -10.7388039, 10.6475525
19: -5.3019552, 4.7420583, -5.2742276, 4.7402177, -7.5772877, 7.5503197
20: -3.4062288, 7.9708629, -3.3956137, 7.9759445, -10.2463646, 10.2626648
21: -1.9484472, 8.8792324, -1.9322827, 8.8783979, -8.9109840, 8.9167366
22: -9.2260237, 2.7835946, -9.2041349, 2.7819586, -8.6785240, 8.6702557
23: 1.3563229, 12.5039062, 1.3716723, 12.4982786, -7.7172184, 7.7011375
24: -2.6836009, 10.4905539, -2.6719577, 10.4816837, -8.1426678, 8.1370697
25: 0.3819842, 13.7513885, 0.3692617, 13.7367134, -9.3331108, 9.3656235
26: -17.4264851, 2.4816518, -17.3786545, 2.4829452, -14.5951118, 14.5457306
27: -10.3208656, 6.2888880, -10.2620230, 6.2897577, -9.1641121, 9.0942898
28: 1.0778267, 13.5767403, 1.1104417, 13.5723028, -9.6303062, 9.5991135
29: -5.1188583, 8.3564167, -5.0831771, 8.3551064, -8.6268902, 8.5865364
30: 5.9857025, 17.7066078, 5.9763136, 17.7008438, -7.6311722, 7.6620770
31: -3.3856714, 10.3919239, -3.3571255, 10.3862085, -9.1543655, 9.1359425
32: -19.5855255, -2.7432547, -19.5696697, -2.7343979, -10.7100639, 10.6797752
33: -47.0340576, -21.5145512, -47.0348053, -21.5634022, -14.4952354, 14.5862007
34: -29.7217426, -10.5717096, -29.7180805, -10.5841370, -10.6645584, 10.6590118
35: -29.2197857, -9.9396877, -29.2198715, -9.9479198, -10.7065277, 10.6907120
36: -31.8921013, -9.3979530, -31.8822479, -9.3856068, -12.7335854, 12.6761551
37: -46.1326523, -23.4802933, -46.1271286, -23.4935684, -16.0618515, 16.0415497
38: -34.3046112, -11.5162907, -34.2978897, -11.4931383, -15.1675568, 15.0852890
39: -56.3195343, -30.6315689, -56.3211212, -30.6727104, -13.1467552, 13.2205315
40: -40.2466965, -23.2933273, -40.2429314, -23.2997856, -8.1793633, 8.1847725
41: -26.7394772, -6.9889126, -26.7250290, -6.9762349, -11.2726517, 11.2191315
42: -14.5374346, -2.0981205, -14.5331459, -2.1203032, -8.5110626, 8.5153198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0718096, upper bound: 5.0912203
time: 23.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0812967, upper bound: 5.0912203
time: 22.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -20.2056236, 0.6745663, -20.2451286, 0.6914787, -13.3953323, 13.4417496
1: -6.4203930, 5.2711015, -6.4483085, 5.2830524, -6.3809853, 6.4233570
2: -11.0086308, 2.2046843, -11.0197592, 2.2416255, -8.5534744, 8.5465546
3: -12.3197060, 3.3981049, -12.3401775, 3.4033990, -11.1280899, 11.1525078
4: -22.0328712, -5.6591969, -22.0491734, -5.6308193, -9.2357407, 9.2450714
5: -10.7834482, 5.6262956, -10.8010187, 5.6672759, -12.1637573, 12.1504898
6: -22.4802456, -4.4985704, -22.4880123, -4.4284935, -11.2180290, 11.1457253
7: -9.5391636, 8.8994370, -9.5660124, 8.9201832, -12.4221992, 12.4804268
8: -26.3327618, -5.6263247, -26.3632298, -5.6029625, -9.7876892, 9.8297501
9: -14.5714808, 2.1667883, -14.5999861, 2.1408474, -12.8859482, 12.9689789
10: -5.8889179, 11.7320271, -5.9215956, 11.7228527, -13.2024345, 13.2604218
11: 9.6109600, 21.1020203, 9.5531988, 21.1099930, -7.4548092, 7.5106430
12: -15.1251860, 9.8281593, -15.1603699, 9.8709621, -18.4931946, 18.4099960
13: -28.0225143, -3.0170429, -28.0428677, -3.0668192, -12.8501892, 12.9490623
14: -31.3659210, 0.6400766, -31.4072647, 0.6531682, -21.4783401, 21.5217285
15: -24.9116631, -10.7107706, -24.9399834, -10.6934605, -8.8629074, 8.8737774
16: -6.9403205, 7.8962240, -6.9573035, 7.8908114, -10.0967102, 10.1388016
17: -14.7552280, 11.7219572, -14.7670879, 11.7472162, -21.7151031, 21.6856537
18: -0.9119453, 12.5626106, -0.8858571, 12.5787144, -10.7156830, 10.6556129
19: -5.2939301, 4.7352247, -5.2877698, 4.7421246, -7.5717525, 7.5587559
20: -3.3908482, 7.9614592, -3.4027190, 7.9878001, -10.2452774, 10.2586403
21: -1.9392502, 8.8751612, -1.9459975, 8.8847475, -8.9058151, 8.9222469
22: -9.2078075, 2.7649097, -9.2054996, 2.7842324, -8.6801701, 8.6496544
23: 1.3703656, 12.5032825, 1.3407538, 12.5084333, -7.7108231, 7.7355080
24: -2.6770327, 10.4943161, -2.7094655, 10.4971323, -8.1467018, 8.1823082
25: 0.3950717, 13.7544060, 0.3118196, 13.7629471, -9.3419991, 9.4303398
26: -17.3953438, 2.4665365, -17.3877525, 2.4843590, -14.5738602, 14.5441589
27: -10.3115797, 6.2871170, -10.2631187, 6.2976890, -9.1599541, 9.0963306
28: 1.0904357, 13.5773230, 1.0839548, 13.5814495, -9.6175880, 9.6283531
29: -5.1011133, 8.3426342, -5.0884695, 8.3574715, -8.6193867, 8.5870247
30: 6.0060744, 17.7094574, 5.9476771, 17.7160625, -7.6310234, 7.6987209
31: -3.3754733, 10.3832960, -3.3781846, 10.3927355, -9.1482544, 9.1476402
32: -19.5864220, -2.7506230, -19.5920258, -2.6967299, -10.7543602, 10.6898804
33: -47.0063286, -21.5575333, -47.0280151, -21.5555058, -14.5140800, 14.5385208
34: -29.7026386, -10.5901661, -29.7151031, -10.5772305, -10.6792145, 10.6387329
35: -29.2086639, -9.9621506, -29.2204132, -9.9424314, -10.7219810, 10.6667366
36: -31.8779335, -9.4017792, -31.8811951, -9.3755245, -12.7436218, 12.6754799
37: -46.1094398, -23.4993057, -46.1274185, -23.4944496, -16.0458603, 16.0279388
38: -34.2944794, -11.5231466, -34.3043137, -11.4900866, -15.1498642, 15.0909119
39: -56.2913857, -30.6598759, -56.3119736, -30.6703472, -13.1582336, 13.1890068
40: -40.2374496, -23.3035069, -40.2486115, -23.2869587, -8.1794205, 8.1838245
41: -26.7363548, -6.9963026, -26.7396717, -6.9448013, -11.2919388, 11.2275238
42: -14.5320759, -2.1048150, -14.5385008, -2.1161995, -8.4922771, 8.5179405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0817854, upper bound: 5.0819536
time: 21.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0912654, upper bound: 5.0819536
time: 58.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.2156639, 0.6924238, -20.2455559, 0.7007620, -13.4158783, 13.4454765
1: -6.4256749, 5.2783012, -6.4485788, 5.2872033, -6.3951340, 6.4196911
2: -11.0166340, 2.2172961, -11.0200453, 2.2491202, -8.5725937, 8.5441189
3: -12.3259172, 3.4067879, -12.3405485, 3.4086859, -11.1457748, 11.1495247
4: -22.0360069, -5.6520009, -22.0491638, -5.6268997, -9.2407837, 9.2457695
5: -10.7930222, 5.6420145, -10.8013000, 5.6764522, -12.1858902, 12.1490097
6: -22.4852028, -4.4987135, -22.4896049, -4.4285784, -11.2254257, 11.1466522
7: -9.5544596, 8.9154291, -9.5672607, 8.9297218, -12.4585953, 12.4743385
8: -26.3426018, -5.6170750, -26.3639317, -5.5972805, -9.8158722, 9.8201637
9: -14.5764704, 2.1720510, -14.6029196, 2.1419511, -12.8859787, 12.9795113
10: -5.8930721, 11.7365046, -5.9236851, 11.7239628, -13.2064552, 13.2679596
11: 9.5884285, 21.1068306, 9.5510902, 21.1129093, -7.4793282, 7.5086479
12: -15.1380033, 9.8351908, -15.1674891, 9.8713694, -18.4909668, 18.4346695
13: -28.0221786, -3.0147388, -28.0426750, -3.0652814, -12.8586502, 12.9484749
14: -31.3747730, 0.6539626, -31.4082031, 0.6614249, -21.4953003, 21.5328674
15: -24.9253387, -10.6888046, -24.9481812, -10.6921940, -8.8592987, 8.8981342
16: -6.9449396, 7.8970375, -6.9581490, 7.8909554, -10.1050148, 10.1396370
17: -14.7625179, 11.7350960, -14.7672653, 11.7544470, -21.7243500, 21.6902313
18: -0.9112239, 12.5643272, -0.8864434, 12.5791798, -10.7181740, 10.6577377
19: -5.2971106, 4.7380624, -5.2885466, 4.7433071, -7.5770397, 7.5620842
20: -3.3975589, 7.9634366, -3.4041362, 7.9889526, -10.2562103, 10.2625885
21: -1.9425040, 8.8751030, -1.9470254, 8.8847904, -8.9126015, 8.9221497
22: -9.2188663, 2.7846484, -9.2116632, 2.7851136, -8.6767159, 8.6733685
23: 1.3643820, 12.5075111, 1.3399744, 12.5104132, -7.7180176, 7.7380714
24: -2.6775339, 10.4959335, -2.7098973, 10.4975929, -8.1475792, 8.1850033
25: 0.3911150, 13.7618017, 0.3097785, 13.7635670, -9.3431244, 9.4372520
26: -17.4062195, 2.4730070, -17.3937149, 2.4846780, -14.5738678, 14.5589142
27: -10.3126984, 6.2879267, -10.2641172, 6.2980862, -9.1645622, 9.0977154
28: 1.0871098, 13.5779848, 1.0826957, 13.5815325, -9.6279526, 9.6289978
29: -5.1064234, 8.3521080, -5.0915442, 8.3579388, -8.6182499, 8.5966759
30: 5.9993525, 17.7128658, 5.9450779, 17.7161694, -7.6383305, 7.6997528
31: -3.3795753, 10.3884869, -3.3787582, 10.3951311, -9.1559906, 9.1509876
32: -19.5906944, -2.7503633, -19.5930557, -2.6965799, -10.7583847, 10.6919956
33: -47.0232925, -21.5376720, -47.0375519, -21.5542297, -14.5123138, 14.5708847
34: -29.7114429, -10.5834751, -29.7203369, -10.5768108, -10.6833458, 10.6495171
35: -29.2202873, -9.9491920, -29.2272682, -9.9419632, -10.7158089, 10.6922150
36: -31.8839340, -9.3978119, -31.8848038, -9.3753185, -12.7413483, 12.6853447
37: -46.1252899, -23.4886169, -46.1368332, -23.4937801, -16.0410080, 16.0612564
38: -34.2963562, -11.5177021, -34.3046761, -11.4869957, -15.1550217, 15.0934715
39: -56.3068466, -30.6474686, -56.3217850, -30.6692429, -13.1578484, 13.2080345
40: -40.2484970, -23.2987595, -40.2540283, -23.2869644, -8.1818466, 8.1947155
41: -26.7426796, -6.9942913, -26.7426834, -6.9445677, -11.2986679, 11.2329369
42: -14.5384636, -2.1026645, -14.5409479, -2.1159582, -8.4971390, 8.5243454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0817854, upper bound: 5.0871884
time: 28.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0912654, upper bound: 5.0871884
time: 21.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.2243900, 0.6982934, -20.2301731, 0.6842349, -13.4069061, 13.4327621
1: -6.4281397, 5.2747040, -6.4295883, 5.2695780, -6.3833981, 6.3969078
2: -11.0191879, 2.2283642, -11.0149269, 2.2388892, -8.5635757, 8.5519924
3: -12.3387966, 3.4239655, -12.3432465, 3.4051094, -11.1503906, 11.1689682
4: -22.0478973, -5.6396103, -22.0495262, -5.6309948, -9.2455940, 9.2547646
5: -10.7972097, 5.6556644, -10.7966118, 5.6623082, -12.1766815, 12.1612587
6: -22.4711933, -4.4980240, -22.4511566, -4.4709187, -11.1798782, 11.1243286
7: -9.5649567, 8.9246111, -9.5661898, 8.9230347, -12.4622574, 12.4815598
8: -26.3561707, -5.6134615, -26.3500023, -5.6137938, -9.8093605, 9.8096085
9: -14.5870962, 2.1818552, -14.5926476, 2.1302500, -12.8798370, 12.9726639
10: -5.8968163, 11.7397919, -5.9036169, 11.7047577, -13.1939926, 13.2470894
11: 9.5732775, 21.1037636, 9.5688772, 21.0996666, -7.4836216, 7.4779854
12: -15.1589508, 9.8583775, -15.1539612, 9.8666573, -18.5070496, 18.4453735
13: -28.0233574, -3.0112402, -28.0348091, -3.0784388, -12.8514061, 12.9488373
14: -31.3996983, 0.6494255, -31.3821869, 0.6432338, -21.5094910, 21.4892273
15: -24.9217148, -10.6927872, -24.9308548, -10.7026367, -8.8502426, 8.8912849
16: -6.9480815, 7.9030638, -6.9481492, 7.8827844, -10.1001396, 10.1352425
17: -14.7878819, 11.7582397, -14.7595501, 11.7646523, -21.7570343, 21.6994781
18: -0.9300289, 12.5724211, -0.8806179, 12.5790586, -10.7361107, 10.6526375
19: -5.3014798, 4.7422547, -5.2848530, 4.7430406, -7.5797024, 7.5638371
20: -3.4033284, 7.9702044, -3.3986697, 7.9792342, -10.2511902, 10.2687798
21: -1.9484804, 8.8778200, -1.9402201, 8.8803253, -8.9149933, 8.9239044
22: -9.2240582, 2.7840009, -9.2050686, 2.7840283, -8.6803894, 8.6721783
23: 1.3565152, 12.5054617, 1.3525681, 12.5027561, -7.7224026, 7.7227631
24: -2.6833830, 10.4854422, -2.6845708, 10.4757576, -8.1374664, 8.1489677
25: 0.3818069, 13.7492695, 0.3416407, 13.7368040, -9.3342247, 9.3927078
26: -17.4255943, 2.4829652, -17.3905640, 2.4894612, -14.6000671, 14.5597992
27: -10.3183661, 6.2884846, -10.2581291, 6.2962089, -9.1626682, 9.0853386
28: 1.0791302, 13.5780087, 1.0907612, 13.5760298, -9.6330643, 9.6202278
29: -5.1166210, 8.3554754, -5.0846720, 8.3556404, -8.6239319, 8.5872192
30: 5.9851365, 17.6998634, 5.9719949, 17.6921310, -7.6277103, 7.6607666
31: -3.3864708, 10.3941250, -3.3779764, 10.3957767, -9.1609612, 9.1578789
32: -19.5723820, -2.7433734, -19.5507469, -2.7391047, -10.6978340, 10.6685600
33: -47.0290947, -21.5152931, -47.0311356, -21.5701447, -14.4853325, 14.5890846
34: -29.7147980, -10.5713844, -29.7070427, -10.5888453, -10.6574097, 10.6505051
35: -29.2098866, -9.9396019, -29.2055531, -9.9620428, -10.6854401, 10.6835747
36: -31.8683853, -9.3976479, -31.8438969, -9.4123096, -12.6940269, 12.6565247
37: -46.1324387, -23.4801025, -46.1391335, -23.4919834, -16.0664444, 16.0639191
38: -34.2874146, -11.5161476, -34.2712173, -11.5208025, -15.1245804, 15.0724182
39: -56.3119011, -30.6321754, -56.3129272, -30.6865768, -13.1256866, 13.2193146
40: -40.2466011, -23.2932816, -40.2452393, -23.2971497, -8.1821671, 8.1868973
41: -26.7291641, -6.9887094, -26.7094803, -6.9727745, -11.2655239, 11.2083206
42: -14.5360374, -2.0982127, -14.5353441, -2.1217752, -8.5073452, 8.5190506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0729187, upper bound: 5.0912655
time: 19.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0824040, upper bound: 5.0912655
time: 18.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.2147961, 0.6896589, -20.2497597, 0.6920021, -13.3975716, 13.4608841
1: -6.4229851, 5.2770777, -6.4494123, 5.2832241, -6.3816395, 6.4303379
2: -11.0163574, 2.2173717, -11.0240583, 2.2421496, -8.5588074, 8.5628548
3: -12.3337193, 3.4156680, -12.3477602, 3.4041624, -11.1368256, 11.1773567
4: -22.0456352, -5.6448073, -22.0561466, -5.6302242, -9.2445412, 9.2667656
5: -10.7923689, 5.6408052, -10.8057985, 5.6679688, -12.1735153, 12.1703491
6: -22.4886570, -4.4974566, -22.4890175, -4.4294777, -11.2358818, 11.1509628
7: -9.5495329, 8.9117537, -9.5708265, 8.9205923, -12.4319305, 12.4969902
8: -26.3467236, -5.6134043, -26.3709641, -5.6027975, -9.7929459, 9.8499336
9: -14.5832138, 2.1826487, -14.6057491, 2.1421516, -12.8916931, 12.9908562
10: -5.8934951, 11.7455807, -5.9224091, 11.7244511, -13.2069473, 13.2778702
11: 9.5952511, 21.1084023, 9.5522118, 21.1134453, -7.4702740, 7.5089836
12: -15.1528034, 9.8514462, -15.1613359, 9.8838387, -18.5337982, 18.4314194
13: -28.0278549, -3.0117259, -28.0432243, -3.0647571, -12.8565216, 12.9597588
14: -31.3916836, 0.6477642, -31.4101830, 0.6569834, -21.5099487, 21.5224457
15: -24.9086723, -10.7089272, -24.9374580, -10.6925983, -8.8629570, 8.8937225
16: -6.9439440, 7.9063706, -6.9584579, 7.8911057, -10.0974617, 10.1505051
17: -14.7817221, 11.7460289, -14.7678795, 11.7601452, -21.7540741, 21.7031708
18: -0.9316549, 12.5718527, -0.8876972, 12.5833960, -10.7441368, 10.6604805
19: -5.2994032, 4.7399430, -5.2887592, 4.7422786, -7.5765057, 7.5661430
20: -3.3994730, 7.9691324, -3.4046905, 7.9882331, -10.2540169, 10.2712250
21: -1.9462368, 8.8802700, -1.9474437, 8.8849249, -8.9122963, 8.9330082
22: -9.2155113, 2.7649355, -9.2059355, 2.7848783, -8.6887398, 8.6563110
23: 1.3618199, 12.5057411, 1.3398008, 12.5084887, -7.7191830, 7.7384033
24: -2.6834674, 10.4965076, -2.7101326, 10.4975147, -8.1538162, 8.1855850
25: 0.3853426, 13.7576437, 0.3109016, 13.7633133, -9.3523769, 9.4341736
26: -17.4165611, 2.4754648, -17.3890285, 2.4892321, -14.6024208, 14.5503082
27: -10.3192749, 6.2887745, -10.2648096, 6.2978153, -9.1722641, 9.0995235
28: 1.0804458, 13.5805864, 1.0828090, 13.5815344, -9.6277771, 9.6329308
29: -5.1140318, 8.3480911, -5.0889349, 8.3597307, -8.6336613, 8.5879097
30: 5.9915881, 17.7101688, 5.9470463, 17.7159710, -7.6361122, 7.7012730
31: -3.3823950, 10.3903427, -3.3802094, 10.3928757, -9.1553154, 9.1599808
32: -19.5923386, -2.7432616, -19.5931587, -2.6958437, -10.7616997, 10.6979961
33: -47.0189247, -21.5336952, -47.0342560, -21.5539589, -14.5098267, 14.5656052
34: -29.7160492, -10.5779905, -29.7215614, -10.5766459, -10.6762123, 10.6560402
35: -29.2099266, -9.9522028, -29.2198944, -9.9417191, -10.7256393, 10.6720657
36: -31.8860111, -9.4013138, -31.8821640, -9.3749962, -12.7580986, 12.6752396
37: -46.1180573, -23.4923935, -46.1288376, -23.4918995, -16.0764313, 16.0332260
38: -34.3046112, -11.5215054, -34.3062057, -11.4896727, -15.1734467, 15.0923004
39: -56.3053703, -30.6438580, -56.3194580, -30.6694794, -13.1538315, 13.2095947
40: -40.2409401, -23.2979679, -40.2500153, -23.2866611, -8.1953049, 8.1834450
41: -26.7418022, -6.9904556, -26.7406864, -6.9442363, -11.3070984, 11.2294846
42: -14.5324678, -2.1000004, -14.5393105, -2.1154766, -8.5129414, 8.5181198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0817854, upper bound: 5.0860607
time: 19.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0912654, upper bound: 5.0860607
time: 14.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2247581, 0.7075377, -20.2502289, 0.7012634, -13.4181366, 13.4646111
1: -6.4282451, 5.2842846, -6.4496880, 5.2873759, -6.3958054, 6.4266701
2: -11.0243711, 2.2300150, -11.0243855, 2.2496293, -8.5779533, 8.5603943
3: -12.3399105, 3.4243832, -12.3481798, 3.4094481, -11.1545410, 11.1743813
4: -22.0487766, -5.6376104, -22.0561218, -5.6262903, -9.2495842, 9.2674637
5: -10.8019390, 5.6565170, -10.8060513, 5.6771231, -12.1956482, 12.1688614
6: -22.4935780, -4.4976234, -22.4906349, -4.4295545, -11.2432861, 11.1519089
7: -9.5648251, 8.9277344, -9.5721016, 8.9302158, -12.4683380, 12.4909096
8: -26.3565826, -5.6041412, -26.3716278, -5.5971165, -9.8211174, 9.8403282
9: -14.5881701, 2.1879077, -14.6087246, 2.1432896, -12.8917236, 13.0014191
10: -5.8976326, 11.7500963, -5.9245267, 11.7255840, -13.2109413, 13.2854004
11: 9.5727005, 21.1132050, 9.5500860, 21.1163712, -7.4947834, 7.5069790
12: -15.1656036, 9.8584461, -15.1685066, 9.8842735, -18.5315323, 18.4560623
13: -28.0276012, -3.0094650, -28.0429764, -3.0632973, -12.8649902, 12.9591789
14: -31.4004974, 0.6616354, -31.4110641, 0.6652856, -21.5268631, 21.5335693
15: -24.9223175, -10.6869421, -24.9456635, -10.6913805, -8.8593559, 8.9181175
16: -6.9485641, 7.9071875, -6.9592943, 7.8912587, -10.1057701, 10.1513424
17: -14.7890759, 11.7590599, -14.7680702, 11.7674313, -21.7633057, 21.7077332
18: -0.9309604, 12.5735550, -0.8883240, 12.5838470, -10.7466202, 10.6625977
19: -5.3025999, 4.7427773, -5.2895393, 4.7434373, -7.5817871, 7.5694733
20: -3.4062128, 7.9711266, -3.4061241, 7.9893751, -10.2649384, 10.2751770
21: -1.9494987, 8.8802204, -1.9484810, 8.8849688, -8.9190979, 8.9329033
22: -9.2265844, 2.7846599, -9.2121077, 2.7857800, -8.6852970, 8.6800327
23: 1.3558506, 12.5099983, 1.3390446, 12.5104771, -7.7263927, 7.7409439
24: -2.6839607, 10.4981079, -2.7105715, 10.4979744, -8.1546936, 8.1882687
25: 0.3813686, 13.7650175, 0.3088841, 13.7639141, -9.3535004, 9.4410782
26: -17.4273834, 2.4819608, -17.3949547, 2.4895437, -14.6024513, 14.5650787
27: -10.3203859, 6.2895994, -10.2657986, 6.2981892, -9.1768703, 9.1008968
28: 1.0771391, 13.5812654, 1.0814931, 13.5816240, -9.6381454, 9.6335793
29: -5.1193600, 8.3575821, -5.0920429, 8.3602428, -8.6325397, 8.5975609
30: 5.9848566, 17.7135830, 5.9444408, 17.7160778, -7.6434193, 7.7023125
31: -3.3865516, 10.3955441, -3.3808012, 10.3952808, -9.1630325, 9.1633415
32: -19.5966034, -2.7429969, -19.5941696, -2.6956804, -10.7657394, 10.7000885
33: -47.0359192, -21.5137749, -47.0437965, -21.5526695, -14.5080528, 14.5979500
34: -29.7248783, -10.5712624, -29.7268085, -10.5762491, -10.6803398, 10.6668129
35: -29.2215080, -9.9392548, -29.2267342, -9.9412346, -10.7194901, 10.6975861
36: -31.8920269, -9.3973675, -31.8857803, -9.3747139, -12.7558556, 12.6851006
37: -46.1338730, -23.4817657, -46.1382675, -23.4913082, -16.0715790, 16.0665817
38: -34.3064194, -11.5159731, -34.3065758, -11.4866629, -15.1786118, 15.0949135
39: -56.3208160, -30.6314850, -56.3292427, -30.6683903, -13.1534195, 13.2286415
40: -40.2519760, -23.2932453, -40.2554245, -23.2866364, -8.1977158, 8.1943531
41: -26.7481918, -6.9885044, -26.7436028, -6.9439654, -11.3138123, 11.2348938
42: -14.5388451, -2.0978603, -14.5417261, -2.1152258, -8.5178032, 8.5245209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1741

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0817854, upper bound: 5.0912655
time: 15.15 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0912654, upper bound: 5.0912655
time: 22.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 40.01 seconds
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0629586, upper bound: 5.0912202
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0724463, upper bound: 5.0912203
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0718096, upper bound: 5.0912203
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0812967, upper bound: 5.0912203
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0817854, upper bound: 5.0819536
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0912654, upper bound: 5.0819536
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0817854, upper bound: 5.0871884
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0912654, upper bound: 5.0871884
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0729187, upper bound: 5.0912655
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0824040, upper bound: 5.0912655
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0817854, upper bound: 5.0860607
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0912654, upper bound: 5.0860607
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0817854, upper bound: 5.0912655
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 40.01
Output dim: 11, lower bound: -5.0912654, upper bound: 5.0912655

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -20.2120113, 0.6995294, -20.1976089, 0.6773448, -13.3784027, 13.3812027
1: -6.4210639, 5.2757559, -6.4086990, 5.2641578, -6.3655567, 6.3626518
2: -11.0103769, 2.2277861, -10.9953680, 2.2165751, -8.5318947, 8.5350838
3: -12.3279533, 3.4236202, -12.3169823, 3.3899226, -11.1244049, 11.1466522
4: -22.0352268, -5.6403475, -22.0228844, -5.6507311, -9.2129402, 9.2327690
5: -10.7843208, 5.6549931, -10.7690687, 5.6348715, -12.1373978, 12.1381798
6: -22.4576302, -4.4993191, -22.4224892, -4.5178699, -11.1188431, 11.0997620
7: -9.5541992, 8.9233894, -9.5418768, 8.9143524, -12.4446945, 12.4561157
8: -26.3442879, -5.6138401, -26.3219299, -5.6198220, -9.7957306, 9.7751160
9: -14.5770473, 2.1809556, -14.5667820, 2.1209264, -12.8620453, 12.9487762
10: -5.8924508, 11.7383089, -5.8869882, 11.6919632, -13.1780319, 13.2293015
11: 9.5761948, 21.0903187, 9.6064825, 21.0719566, -7.4579649, 7.4242382
12: -15.1570730, 9.8539486, -15.1428070, 9.8502283, -18.4869156, 18.4290009
13: -28.0141239, -3.0152018, -28.0027733, -3.0926800, -12.8281860, 12.9131546
14: -31.3969383, 0.6456285, -31.3456802, 0.6320002, -21.4893723, 21.4354706
15: -24.9207230, -10.6950102, -24.9243698, -10.7137165, -8.8373909, 8.8792000
16: -6.9397678, 7.9034886, -6.9291258, 7.8758097, -10.0828323, 10.1045189
17: -14.7834949, 11.7548323, -14.7268782, 11.7563829, -21.7393265, 21.6577148
18: -0.9291725, 12.5644102, -0.8679473, 12.5578737, -10.7174873, 10.6315231
19: -5.3003273, 4.7377815, -5.2685208, 4.7322478, -7.5687733, 7.5408096
20: -3.4023471, 7.9666409, -3.3861289, 7.9590311, -10.2261009, 10.2512779
21: -1.9464948, 8.8738918, -1.9221518, 8.8677559, -8.9013023, 8.9039001
22: -9.2230816, 2.7770128, -9.1963148, 2.7682819, -8.6612358, 8.6557274
23: 1.3574374, 12.4930458, 1.3860403, 12.4777832, -7.7021732, 7.6769466
24: -2.6826427, 10.4724627, -2.6452637, 10.4484978, -8.1155453, 8.0920277
25: 0.3829510, 13.7315292, 0.4030726, 13.7015476, -9.3049755, 9.3123932
26: -17.4240303, 2.4748306, -17.3729706, 2.4669671, -14.5793457, 14.5324707
27: -10.3182583, 6.2799006, -10.2532701, 6.2718620, -9.1346397, 9.0698509
28: 1.0803471, 13.5673828, 1.1207304, 13.5543861, -9.6146736, 9.5798569
29: -5.1157336, 8.3475666, -5.0750380, 8.3368683, -8.6045914, 8.5689316
30: 5.9868526, 17.6894836, 6.0055680, 17.6699810, -7.6091232, 7.6157856
31: -3.3849220, 10.3858185, -3.3529742, 10.3772869, -9.1440697, 9.1255627
32: -19.5603828, -2.7444913, -19.5244026, -2.7794340, -10.6395073, 10.6455536
33: -47.0255814, -21.5171356, -47.0187988, -21.5829830, -14.4689255, 14.5737648
34: -29.7110710, -10.5728321, -29.6971283, -10.5986347, -10.6393738, 10.6409454
35: -29.2077198, -9.9415436, -29.1978493, -9.9717884, -10.6683197, 10.6739540
36: -31.8681927, -9.4012699, -31.8397331, -9.4291954, -12.6661491, 12.6443596
37: -46.1309090, -23.4796104, -46.1272621, -23.4961357, -16.0545120, 16.0371475
38: -34.2850075, -11.5185699, -34.2612076, -11.5315924, -15.1085663, 15.0595322
39: -56.3050308, -30.6327667, -56.2935371, -30.6919193, -13.1139259, 13.2031555
40: -40.2386856, -23.2934685, -40.2275314, -23.3104439, -8.1606483, 8.1713047
41: -26.7198524, -6.9912348, -26.6896267, -7.0093036, -11.2202911, 11.1894073
42: -14.5337105, -2.0986342, -14.5249119, -2.1271615, -8.4992657, 8.5076981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0626865, upper bound: 5.0837445
time: 68.15 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0626865, upper bound: 5.0910806
time: 22.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -20.2211800, 0.7000835, -20.2176018, 0.7439914, -13.4590607, 13.3919716
1: -6.4263868, 5.2759223, -6.4204950, 5.3042479, -6.4145699, 6.3687649
2: -11.0137806, 2.2280245, -11.0030956, 2.2473142, -8.5685234, 8.5393410
3: -12.3347492, 3.4235601, -12.3321857, 3.4344220, -11.1737823, 11.1540871
4: -22.0420609, -5.6401777, -22.0374317, -5.6026525, -9.2721024, 9.2402458
5: -10.7901669, 5.6550951, -10.7815723, 5.6790013, -12.1868668, 12.1440659
6: -22.4584465, -4.5004244, -22.4312859, -4.5180230, -11.1218605, 11.1091690
7: -9.5610189, 8.9233704, -9.5590343, 8.9576244, -12.4979172, 12.4674416
8: -26.3525467, -5.6138234, -26.3407688, -5.5720096, -9.8559341, 9.7855186
9: -14.5846424, 2.1810269, -14.5840693, 2.1806972, -12.9255524, 12.9567490
10: -5.8933363, 11.7385674, -5.8924398, 11.7284145, -13.2182388, 13.2344284
11: 9.5749407, 21.0943184, 9.5728989, 21.0808411, -7.4645548, 7.4579506
12: -15.1571617, 9.8547316, -15.1678734, 9.8555794, -18.4922791, 18.4457321
13: -28.0209389, -3.0144703, -28.0174408, -3.0347140, -12.8949356, 12.9210625
14: -31.3958549, 0.6460335, -31.3498344, 0.6498380, -21.5084152, 21.4459763
15: -24.9201469, -10.6939297, -24.9238338, -10.7003365, -8.8531075, 8.8806934
16: -6.9456773, 7.9035978, -6.9430275, 7.9237342, -10.1395149, 10.1136017
17: -14.7860270, 11.7550259, -14.7369061, 11.7747612, -21.7537994, 21.6725922
18: -0.9291756, 12.5701494, -0.9113166, 12.5710230, -10.7240257, 10.6769409
19: -5.3007236, 4.7407093, -5.3036237, 4.7386417, -7.5723972, 7.5775776
20: -3.4031031, 7.9691734, -3.4205916, 7.9650621, -10.2318649, 10.2851906
21: -1.9472129, 8.8761883, -1.9570768, 8.8727150, -8.9051933, 8.9398956
22: -9.2233944, 2.7813847, -9.2316027, 2.7789505, -8.6686916, 8.6962185
23: 1.3571255, 12.4980898, 1.3411973, 12.4884949, -7.7079926, 7.7276230
24: -2.6829076, 10.4767246, -2.6850944, 10.4583797, -8.1216660, 8.1382790
25: 0.3826540, 13.7343082, 0.3701024, 13.7087488, -9.3107719, 9.3513603
26: -17.4244175, 2.4810398, -17.4324570, 2.4804626, -14.5862656, 14.6008377
27: -10.3186674, 6.2861333, -10.3027363, 6.2854657, -9.1419773, 9.1280556
28: 1.0799904, 13.5722313, 1.0705512, 13.5646696, -9.6202965, 9.6331177
29: -5.1159687, 8.3528643, -5.1151171, 8.3494587, -8.6128445, 8.6172791
30: 5.9862103, 17.6920204, 5.9781566, 17.6755371, -7.6138268, 7.6442757
31: -3.3854148, 10.3895063, -3.3932173, 10.3852978, -9.1488495, 9.1682072
32: -19.5610695, -2.7453644, -19.5342598, -2.7792809, -10.6438408, 10.6593132
33: -47.0264130, -21.5164528, -47.0267944, -21.5632782, -14.4790192, 14.5909157
34: -29.7114029, -10.5735855, -29.7095566, -10.5981064, -10.6423721, 10.6574364
35: -29.2078285, -9.9425411, -29.2086372, -9.9697838, -10.6723785, 10.6883049
36: -31.8682499, -9.4017639, -31.8589840, -9.4273767, -12.6693420, 12.6682091
37: -46.1307220, -23.4807663, -46.1313782, -23.4948120, -16.0562363, 16.0450058
38: -34.2850761, -11.5198708, -34.2772217, -11.5324116, -15.1088409, 15.0775146
39: -56.3091164, -30.6324310, -56.3045998, -30.6616402, -13.1387253, 13.2131081
40: -40.2394066, -23.2934341, -40.2322693, -23.3020134, -8.1725502, 8.1761513
41: -26.7203121, -6.9919095, -26.7057266, -7.0069008, -11.2258739, 11.2065353
42: -14.5344200, -2.0985770, -14.5327492, -2.1235230, -8.5035000, 8.5181446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0721745, upper bound: 5.0837445
time: 22.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0721745, upper bound: 5.0910806
time: 8.28 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -20.2123966, 0.7086835, -20.2176514, 0.6943483, -13.3896255, 13.4130173
1: -6.4212084, 5.2853708, -6.4288225, 5.2818995, -6.3779545, 6.3924084
2: -11.0155449, 2.2294793, -11.0048904, 2.2273388, -8.5462456, 8.5435562
3: -12.3290720, 3.4240146, -12.3220291, 3.3942432, -11.1284981, 11.1520996
4: -22.0361156, -5.6383572, -22.0294724, -5.6460357, -9.2169113, 9.2454033
5: -10.7890549, 5.6558361, -10.7785549, 5.6496844, -12.1565399, 12.1458397
6: -22.4800282, -4.4989338, -22.4618416, -4.4765477, -11.1822281, 11.1272850
7: -9.5540466, 8.9265490, -9.5477657, 8.9216452, -12.4507751, 12.4654922
8: -26.3447056, -5.6045132, -26.3433952, -5.6031218, -9.8074455, 9.8058014
9: -14.5781813, 2.1870389, -14.5827723, 2.1340182, -12.8739243, 12.9775200
10: -5.8931961, 11.7486458, -5.9078646, 11.7128410, -13.1950645, 13.2675514
11: 9.5756121, 21.0997429, 9.5877018, 21.0886345, -7.4692020, 7.4532223
12: -15.1636724, 9.8540010, -15.1573362, 9.8677711, -18.5113983, 18.4396362
13: -28.0182381, -3.0134022, -28.0110149, -3.0775146, -12.8417358, 12.9235115
14: -31.3977184, 0.6577759, -31.3745441, 0.6540246, -21.5067444, 21.4797974
15: -24.9213142, -10.6891804, -24.9391327, -10.7024193, -8.8464966, 8.9060574
16: -6.9402514, 7.9075994, -6.9402161, 7.8843384, -10.0884323, 10.1206093
17: -14.7846432, 11.7556705, -14.7354803, 11.7591486, -21.7455978, 21.6660004
18: -0.9301236, 12.5655737, -0.8756382, 12.5626125, -10.7279510, 10.6415215
19: -5.3014503, 4.7383146, -5.2731867, 4.7326589, -7.5708275, 7.5464535
20: -3.4052322, 7.9675589, -3.3936148, 7.9692574, -10.2398567, 10.2576790
21: -1.9475085, 8.8762102, -1.9303851, 8.8723440, -8.9055099, 8.9128132
22: -9.2256546, 2.7776742, -9.2033615, 2.7700262, -8.6661701, 8.6635933
23: 1.3567790, 12.4975739, 1.3725396, 12.4854774, -7.7061386, 7.6951065
24: -2.6832080, 10.4851379, -2.6712084, 10.4707394, -8.1327782, 8.1312847
25: 0.3825238, 13.7472820, 0.3703134, 13.7286043, -9.3242607, 9.3607368
26: -17.4258499, 2.4737813, -17.3774624, 2.4670756, -14.5817490, 14.5376740
27: -10.3202963, 6.2810240, -10.2609062, 6.2738438, -9.1488876, 9.0854492
28: 1.0783253, 13.5706129, 1.1114800, 13.5599403, -9.6197205, 9.5931969
29: -5.1185083, 8.3496456, -5.0824380, 8.3415165, -8.6132374, 8.5792999
30: 5.9865770, 17.7032013, 5.9780140, 17.6939659, -7.6248779, 7.6573296
31: -3.3849840, 10.3872690, -3.3557603, 10.3767843, -9.1461182, 9.1310158
32: -19.5845833, -2.7441201, -19.5677872, -2.7360539, -10.7073708, 10.6770802
33: -47.0323906, -21.5156441, -47.0314178, -21.5655193, -14.4916649, 14.5826645
34: -29.7211056, -10.5726652, -29.7168503, -10.5860453, -10.6622810, 10.6571884
35: -29.2193909, -9.9412260, -29.2190952, -9.9509497, -10.7023888, 10.6881561
36: -31.8918438, -9.4009590, -31.8817139, -9.3917398, -12.7279396, 12.6729240
37: -46.1323471, -23.4811935, -46.1265106, -23.4953690, -16.0596542, 16.0400467
38: -34.3039360, -11.5184565, -34.2965927, -11.4974432, -15.1626434, 15.0820427
39: -56.3139534, -30.6320629, -56.3099289, -30.6736755, -13.1416397, 13.2124519
40: -40.2440948, -23.2933807, -40.2376633, -23.2999306, -8.1762161, 8.1787071
41: -26.7388363, -6.9910202, -26.7237892, -6.9804950, -11.2685738, 11.2159424
42: -14.5364857, -2.0982969, -14.5312862, -2.1206458, -8.5096912, 8.5131855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0715371, upper bound: 5.0837445
time: 19.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0626865, upper bound: 5.0910806
time: 22.50 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20.2216072, 0.7092943, -20.2376347, 0.7609828, -13.4702797, 13.4238052
1: -6.4265156, 5.2855062, -6.4406190, 5.3219643, -6.4269524, 6.3985157
2: -11.0189552, 2.2296555, -11.0126266, 2.2580805, -8.5828590, 8.5478096
3: -12.3359127, 3.4239504, -12.3372011, 3.4387517, -11.1778717, 11.1595345
4: -22.0429382, -5.6381912, -22.0440025, -5.5979524, -9.2760544, 9.2528992
5: -10.7949076, 5.6559715, -10.7910118, 5.6938171, -12.2059860, 12.1517334
6: -22.4808159, -4.5000467, -22.4706135, -4.4766989, -11.1852226, 11.1366806
7: -9.5608959, 8.9265509, -9.5649929, 8.9649124, -12.5039902, 12.4768295
8: -26.3529510, -5.6044984, -26.3623085, -5.5553017, -9.8676567, 9.8161964
9: -14.5857601, 2.1871068, -14.6000557, 2.1937330, -12.9374161, 12.9854736
10: -5.8941107, 11.7488480, -5.9133034, 11.7492809, -13.2352142, 13.2726555
11: 9.5743942, 21.1037560, 9.5541410, 21.0975418, -7.4757643, 7.4869194
12: -15.1637869, 9.8547983, -15.1824474, 9.8731346, -18.5167236, 18.4564438
13: -28.0251122, -3.0127339, -28.0256386, -3.0196428, -12.9084816, 12.9314156
14: -31.3966255, 0.6582263, -31.3787384, 0.6718383, -21.5257797, 21.4902649
15: -24.9207611, -10.6880817, -24.9385834, -10.6890907, -8.8622017, 8.9075356
16: -6.9461365, 7.9077616, -6.9541383, 7.9322453, -10.1451187, 10.1296921
17: -14.7871370, 11.7558765, -14.7454548, 11.7774820, -21.7600479, 21.6809311
18: -0.9300911, 12.5713120, -0.9190087, 12.5757742, -10.7345009, 10.6869431
19: -5.3018303, 4.7412419, -5.3082819, 4.7390342, -7.5744438, 7.5832138
20: -3.4059651, 7.9700856, -3.4280686, 7.9752669, -10.2456245, 10.2915573
21: -1.9482158, 8.8785286, -1.9653165, 8.8773050, -8.9093819, 8.9488068
22: -9.2259197, 2.7820396, -9.2386618, 2.7806966, -8.6736145, 8.7040825
23: 1.3564613, 12.5025940, 1.3276894, 12.4962158, -7.7119522, 7.7457829
24: -2.6835008, 10.4894142, -2.7110336, 10.4806223, -8.1389160, 8.1775360
25: 0.3822150, 13.7500420, 0.3373590, 13.7358036, -9.3300457, 9.3997040
26: -17.4262009, 2.4799833, -17.4369488, 2.4805486, -14.5886879, 14.6060181
27: -10.3207331, 6.2872567, -10.3103447, 6.2874460, -9.1562233, 9.1436520
28: 1.0779757, 13.5754690, 1.0612848, 13.5702477, -9.6253510, 9.6464462
29: -5.1186962, 8.3549871, -5.1225033, 8.3540964, -8.6214943, 8.6276245
30: 5.9859381, 17.7057133, 5.9506159, 17.6995163, -7.6295738, 7.6858215
31: -3.3854599, 10.3909416, -3.3960061, 10.3847799, -9.1508789, 9.1736774
32: -19.5853119, -2.7450178, -19.5776157, -2.7359245, -10.7117119, 10.6908302
33: -47.0332794, -21.5149498, -47.0393829, -21.5458412, -14.5017319, 14.5998001
34: -29.7215061, -10.5734262, -29.7292843, -10.5855799, -10.6652641, 10.6736755
35: -29.2195091, -9.9421940, -29.2299042, -9.9489870, -10.7064323, 10.7024841
36: -31.8919296, -9.4014997, -31.9008789, -9.3898163, -12.7311401, 12.6967773
37: -46.1322174, -23.4823666, -46.1306725, -23.4940929, -16.0613785, 16.0479507
38: -34.3040733, -11.5197430, -34.3126183, -11.4982929, -15.1629028, 15.1000137
39: -56.3180733, -30.6317902, -56.3210449, -30.6434498, -13.1664276, 13.2223740
40: -40.2447815, -23.2933540, -40.2424202, -23.2915058, -8.1881104, 8.1835499
41: -26.7392769, -6.9917111, -26.7398529, -6.9781685, -11.2741356, 11.2330780
42: -14.5372066, -2.0982332, -14.5391150, -2.1170266, -8.5139427, 8.5236149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0810243, upper bound: 5.0837445
time: 15.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0810243, upper bound: 5.0910806
time: 18.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.2029057, 0.6743314, -20.2410583, 0.7565377, -13.4623413, 13.4284973
1: -6.4188547, 5.2710514, -6.4462729, 5.3227301, -6.4225388, 6.4155941
2: -11.0076256, 2.2045956, -11.0186443, 2.2717924, -8.5852928, 8.5420341
3: -12.3178940, 3.3978920, -12.3379221, 3.4476130, -11.1702042, 11.1462936
4: -22.0310287, -5.6594114, -22.0464020, -5.5834198, -9.2859802, 9.2370949
5: -10.7818680, 5.6261959, -10.7985220, 5.7108917, -12.2058868, 12.1435242
6: -22.4799194, -4.5009127, -22.4945145, -4.4311228, -11.2171249, 11.1512833
7: -9.5366364, 8.8992558, -9.5642710, 8.9631166, -12.4657135, 12.4728127
8: -26.3305569, -5.6263452, -26.3609924, -5.5552173, -9.8371468, 9.8191261
9: -14.5694828, 2.1662667, -14.5978508, 2.1994977, -12.9406967, 12.9630547
10: -5.8860207, 11.7315865, -5.9194508, 11.7579002, -13.2382736, 13.2592697
11: 9.6114244, 21.1008987, 9.5230131, 21.1085148, -7.4495773, 7.5346012
12: -15.1248236, 9.8249931, -15.1846027, 9.8683300, -18.4930649, 18.4232712
13: -28.0206299, -3.0172319, -28.0399227, -3.0107229, -12.9066544, 12.9400978
14: -31.3629513, 0.6399450, -31.4076958, 0.6698246, -21.4934387, 21.5257263
15: -24.9103775, -10.7111921, -24.9379883, -10.6831074, -8.8749847, 8.8723030
16: -6.9385643, 7.8961163, -6.9557562, 7.9382515, -10.1450043, 10.1321545
17: -14.7541609, 11.7218704, -14.7698507, 11.7649879, -21.7234497, 21.6913986
18: -0.9115055, 12.5610962, -0.9283628, 12.5772181, -10.7114029, 10.6950035
19: -5.2937789, 4.7344217, -5.3218627, 4.7409592, -7.5689316, 7.5916786
20: -3.3905745, 7.9606876, -3.4352031, 7.9871359, -10.2445374, 10.2875252
21: -1.9390137, 8.8744678, -1.9790556, 8.8836327, -8.9042244, 8.9543190
22: -9.2076960, 2.7633634, -9.2400608, 2.7829671, -8.6752758, 8.6834984
23: 1.3704803, 12.5019598, 1.2967510, 12.5063467, -7.7055492, 7.7801628
24: -2.6769383, 10.4931765, -2.7485886, 10.4960728, -8.1429405, 8.2228012
25: 0.3952830, 13.7530994, 0.2798865, 13.7620678, -9.3389244, 9.4644356
26: -17.3950729, 2.4648676, -17.4460678, 2.4819701, -14.5674019, 14.6045303
27: -10.3114243, 6.2854948, -10.3114710, 6.2954631, -9.1520634, 9.1456947
28: 1.0905750, 13.5760727, 1.0347869, 13.5793953, -9.6126251, 9.6757050
29: -5.1009550, 8.3411760, -5.1278071, 8.3564520, -8.6139851, 8.6281128
30: 6.0063014, 17.7085609, 5.9219847, 17.7147198, -7.6294270, 7.7224846
31: -3.3752992, 10.3822803, -3.4170558, 10.3913336, -9.1447754, 9.1853828
32: -19.5861588, -2.7523384, -19.5999851, -2.6982470, -10.7560043, 10.7009487
33: -47.0055466, -21.5579033, -47.0326309, -21.5379810, -14.5205612, 14.5521202
34: -29.7024384, -10.5919647, -29.7263088, -10.5786505, -10.6799240, 10.6533928
35: -29.2084141, -9.9646597, -29.2304039, -9.9435043, -10.7218971, 10.6785126
36: -31.8777351, -9.4053535, -31.8998909, -9.3797903, -12.7411880, 12.6961174
37: -46.1089478, -23.5013657, -46.1310043, -23.4949455, -16.0453796, 16.0343399
38: -34.2939529, -11.5266418, -34.3190575, -11.4952126, -15.1451950, 15.1055832
39: -56.2899246, -30.6600418, -56.3118210, -30.6410084, -13.1779251, 13.1908493
40: -40.2355652, -23.3035011, -40.2481194, -23.2786636, -8.1881828, 8.1825943
41: -26.7361069, -6.9991274, -26.7545300, -6.9467292, -11.2934532, 11.2414856
42: -14.5318518, -2.1049323, -14.5444870, -2.1129141, -8.4951496, 8.5262203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0745832
time: 31.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0818185
time: 23.30 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20.2128849, 0.6922226, -20.2415199, 0.7657311, -13.4828987, 13.4322166
1: -6.4241452, 5.2782803, -6.4465194, 5.3268785, -6.4366951, 6.4119205
2: -11.0156384, 2.2172134, -11.0189304, 2.2792988, -8.6044083, 8.5395966
3: -12.3240929, 3.4066224, -12.3383522, 3.4528747, -11.1878815, 11.1433372
4: -22.0341492, -5.6521854, -22.0463829, -5.5794630, -9.2910156, 9.2377968
5: -10.7914362, 5.6418505, -10.7987900, 5.7200594, -12.2280121, 12.1420517
6: -22.4848785, -4.5010724, -22.4960938, -4.4312005, -11.2245331, 11.1522369
7: -9.5519133, 8.9152832, -9.5655365, 8.9726858, -12.5021286, 12.4667435
8: -26.3403873, -5.6171107, -26.3616886, -5.5495672, -9.8653259, 9.8095322
9: -14.5744648, 2.1715577, -14.6007853, 2.2005830, -12.9406891, 12.9735794
10: -5.8901443, 11.7361050, -5.9215660, 11.7590551, -13.2423172, 13.2667770
11: 9.5888853, 21.1057072, 9.5209064, 21.1114502, -7.4740944, 7.5326099
12: -15.1376057, 9.8320255, -15.1917620, 9.8687401, -18.4908524, 18.4479218
13: -28.0203400, -3.0150175, -28.0396423, -3.0092530, -12.9150925, 12.9394989
14: -31.3718224, 0.6538014, -31.4086037, 0.6781068, -21.5103607, 21.5368576
15: -24.9240627, -10.6892252, -24.9461784, -10.6818686, -8.8713799, 8.8966675
16: -6.9431529, 7.8969135, -6.9566426, 7.9383688, -10.1532936, 10.1329842
17: -14.7614574, 11.7349520, -14.7700443, 11.7722092, -21.7326584, 21.6959839
18: -0.9107912, 12.5627909, -0.9289854, 12.5776606, -10.7138824, 10.6971283
19: -5.2969732, 4.7372689, -5.3226385, 4.7421265, -7.5742073, 7.5950031
20: -3.3972945, 7.9626713, -3.4366174, 7.9882669, -10.2554626, 10.2914734
21: -1.9422641, 8.8744087, -1.9800986, 8.8836937, -8.9110222, 8.9542007
22: -9.2187614, 2.7831151, -9.2462072, 2.7838683, -8.6718407, 8.7071857
23: 1.3645138, 12.5061970, 1.2959461, 12.5083570, -7.7127552, 7.7827129
24: -2.6774378, 10.4948025, -2.7489984, 10.4965229, -8.1438255, 8.2254982
25: 0.3913147, 13.7604828, 0.2778678, 13.7626705, -9.3400650, 9.4713287
26: -17.4059620, 2.4713471, -17.4520035, 2.4822831, -14.5674324, 14.6192780
27: -10.3125458, 6.2862816, -10.3124313, 6.2958040, -9.1566563, 9.1470718
28: 1.0872726, 13.5767202, 1.0335138, 13.5794849, -9.6229935, 9.6763573
29: -5.1063023, 8.3506851, -5.1309299, 8.3569508, -8.6128521, 8.6377602
30: 5.9995737, 17.7119465, 5.9193907, 17.7148285, -7.6367188, 7.7235069
31: -3.3794162, 10.3874893, -3.4176414, 10.3937330, -9.1525116, 9.1887436
32: -19.5904121, -2.7520797, -19.6010036, -2.6980867, -10.7600288, 10.7030506
33: -47.0224648, -21.5380192, -47.0421486, -21.5367012, -14.5188103, 14.5844536
34: -29.7112198, -10.5852222, -29.7315273, -10.5782766, -10.6840630, 10.6641960
35: -29.2200050, -9.9517288, -29.2373085, -9.9429951, -10.7157249, 10.7040100
36: -31.8837166, -9.4014482, -31.9034576, -9.3795538, -12.7389412, 12.7059708
37: -46.1248093, -23.4906921, -46.1404305, -23.4943104, -16.0405197, 16.0676956
38: -34.2957802, -11.5211830, -34.3194237, -11.4921799, -15.1503525, 15.1081848
39: -56.3054008, -30.6476307, -56.3216400, -30.6399231, -13.1775398, 13.2099037
40: -40.2465858, -23.2987995, -40.2535095, -23.2786312, -8.1905975, 8.1935043
41: -26.7424889, -6.9971094, -26.7575130, -6.9464564, -11.3001766, 11.2468910
42: -14.5382137, -2.1027970, -14.5468874, -2.1126480, -8.5000114, 8.5326271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0797499
time: 27.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0870523
time: 16.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -20.2124386, 0.6975229, -20.2061424, 0.6826208, -13.3932915, 13.4087143
1: -6.4212561, 5.2745275, -6.4157438, 5.2692022, -6.3759727, 6.3830280
2: -11.0147429, 2.2280602, -11.0060863, 2.2382863, -8.5587540, 8.5432339
3: -12.3301754, 3.4238238, -12.3258600, 3.4047637, -11.1431122, 11.1553345
4: -22.0392151, -5.6399198, -22.0322227, -5.6316504, -9.2366867, 9.2392921
5: -10.7897768, 5.6553926, -10.7816133, 5.6617203, -12.1693268, 12.1484070
6: -22.4700832, -4.4992790, -22.4489326, -4.4733825, -11.1759605, 11.1205063
7: -9.5555992, 8.9244823, -9.5472775, 8.9227552, -12.4524841, 12.4625969
8: -26.3457260, -5.6135292, -26.3289547, -5.6139135, -9.7986031, 9.7885933
9: -14.5774403, 2.1812820, -14.5732107, 2.1291571, -12.8711014, 12.9588013
10: -5.8930230, 11.7391109, -5.8960361, 11.7034035, -13.1896782, 13.2407570
11: 9.5749722, 21.0986347, 9.5722733, 21.0892963, -7.4717960, 7.4682369
12: -15.1585197, 9.8543968, -15.1531181, 9.8586788, -18.5015182, 18.4418716
13: -28.0146828, -3.0121331, -28.0171547, -3.0802612, -12.8411255, 12.9319763
14: -31.3978348, 0.6489601, -31.3784332, 0.6420932, -21.5055237, 21.4827652
15: -24.9209843, -10.6942739, -24.9293766, -10.7056704, -8.8465958, 8.8883057
16: -6.9404244, 7.9028597, -6.9327064, 7.8822813, -10.0917664, 10.1194878
17: -14.7842407, 11.7579813, -14.7522755, 11.7640953, -21.7509842, 21.6903534
18: -0.9295924, 12.5651379, -0.8797698, 12.5644045, -10.7252808, 10.6465797
19: -5.3009768, 4.7385283, -5.2838278, 4.7354608, -7.5732574, 7.5599499
20: -3.4023199, 7.9668899, -3.3966482, 7.9725447, -10.2447128, 10.2637634
21: -1.9475459, 8.8748245, -1.9383287, 8.8742504, -8.9095230, 8.9199791
22: -9.2236719, 2.7780914, -9.2043400, 2.7720976, -8.6680393, 8.6655197
23: 1.3569477, 12.4991131, 1.3534051, 12.4899454, -7.7113132, 7.7167168
24: -2.6830204, 10.4800186, -2.6838105, 10.4647961, -8.1275845, 8.1431923
25: 0.3823483, 13.7451572, 0.3427103, 13.7287130, -9.3253899, 9.3878326
26: -17.4249725, 2.4751108, -17.3893509, 2.4735932, -14.5867119, 14.5517578
27: -10.3177471, 6.2806215, -10.2570000, 6.2802796, -9.1474152, 9.0764923
28: 1.0796516, 13.5718880, 1.0917878, 13.5636816, -9.6224747, 9.6142998
29: -5.1162620, 8.3487320, -5.0839434, 8.3420515, -8.6102619, 8.5799866
30: 5.9860048, 17.6964512, 5.9737253, 17.6852169, -7.6214218, 7.6560249
31: -3.3857923, 10.3894606, -3.3766022, 10.3863573, -9.1526985, 9.1529541
32: -19.5713959, -2.7441876, -19.5488663, -2.7407513, -10.6951408, 10.6658478
33: -47.0274582, -21.5163441, -47.0277863, -21.5722790, -14.4817238, 14.5855217
34: -29.7141743, -10.5723934, -29.7057915, -10.5907326, -10.6551208, 10.6486855
35: -29.2094555, -9.9411545, -29.2047443, -9.9650955, -10.6812782, 10.6810341
36: -31.8681374, -9.4006844, -31.8432388, -9.4184132, -12.6883774, 12.6532631
37: -46.1321030, -23.4810486, -46.1386032, -23.4937878, -16.0641785, 16.0624695
38: -34.2867889, -11.5183144, -34.2698708, -11.5251074, -15.1195908, 15.0691605
39: -56.3063202, -30.6326294, -56.3017044, -30.6875153, -13.1205864, 13.2112389
40: -40.2440033, -23.2933559, -40.2399979, -23.2972736, -8.1790047, 8.1808357
41: -26.7285633, -6.9908180, -26.7081947, -6.9770193, -11.2614441, 11.2051582
42: -14.5351067, -2.0983748, -14.5334644, -2.1221256, -8.5059738, 8.5169067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0727759, upper bound: 5.0837803
time: 17.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0727807, upper bound: 5.0911285
time: 17.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -20.2216358, 0.6980779, -20.2260857, 0.7492921, -13.4739151, 13.4195023
1: -6.4265642, 5.2746487, -6.4275246, 5.3092632, -6.4249573, 6.3891411
2: -11.0181847, 2.2282591, -11.0138226, 2.2690306, -8.5953941, 8.5474739
3: -12.3369799, 3.4237614, -12.3410330, 3.4492731, -11.1925049, 11.1627502
4: -22.0460606, -5.6397448, -22.0467510, -5.5835781, -9.2958488, 9.2467880
5: -10.7956305, 5.6554918, -10.7941170, 5.7058592, -12.2187958, 12.1543159
6: -22.4708519, -4.5003920, -22.4577198, -4.4735451, -11.1789703, 11.1299095
7: -9.5624523, 8.9244156, -9.5644531, 8.9660416, -12.5057373, 12.4739304
8: -26.3539257, -5.6135135, -26.3478031, -5.5660682, -9.8588181, 9.7989807
9: -14.5850735, 2.1813383, -14.5905495, 2.1888647, -12.9345856, 12.9667511
10: -5.8939309, 11.7393675, -5.9014506, 11.7398567, -13.2298660, 13.2458916
11: 9.5737123, 21.1026363, 9.5387020, 21.0982132, -7.4783821, 7.5019417
12: -15.1586380, 9.8552094, -15.1782246, 9.8640490, -18.5068436, 18.4586258
13: -28.0215111, -3.0114460, -28.0317879, -3.0222998, -12.9078712, 12.9398651
14: -31.3967628, 0.6492739, -31.3825817, 0.6599078, -21.5245361, 21.4932404
15: -24.9204140, -10.6932240, -24.9288559, -10.6922865, -8.8623085, 8.8898125
16: -6.9463263, 7.9029784, -6.9466224, 7.9302325, -10.1484070, 10.1285801
17: -14.7867136, 11.7581902, -14.7622929, 11.7824287, -21.7654037, 21.7052689
18: -0.9295690, 12.5709076, -0.9231575, 12.5775471, -10.7318268, 10.6920204
19: -5.3013515, 4.7414417, -5.3189402, 4.7418389, -7.5768738, 7.5967255
20: -3.4030647, 7.9694223, -3.4311166, 7.9785490, -10.2504730, 10.2976608
21: -1.9482604, 8.8771324, -1.9732788, 8.8792152, -8.9134064, 8.9559803
22: -9.2239532, 2.7824612, -9.2396164, 2.7827721, -8.6754837, 8.7060127
23: 1.3566153, 12.5041342, 1.3085740, 12.5006752, -7.7171326, 7.7674122
24: -2.6832836, 10.4842796, -2.7236652, 10.4746780, -8.1337070, 8.1894531
25: 0.3820238, 13.7479458, 0.3097432, 13.7359238, -9.3311691, 9.4267998
26: -17.4253235, 2.4812894, -17.4488239, 2.4870739, -14.5936432, 14.6201324
27: -10.3182087, 6.2868457, -10.3065138, 6.2939062, -9.1547661, 9.1346874
28: 1.0792890, 13.5767603, 1.0416005, 13.5739727, -9.6280975, 9.6675682
29: -5.1164427, 8.3540297, -5.1240330, 8.3546190, -8.6185207, 8.6283226
30: 5.9853354, 17.6989975, 5.9462881, 17.6907997, -7.6261005, 7.6845226
31: -3.3862906, 10.3931198, -3.4168689, 10.3943520, -9.1574669, 9.1956100
32: -19.5720825, -2.7450678, -19.5587215, -2.7406023, -10.6994705, 10.6796112
33: -47.0282936, -21.5156765, -47.0356979, -21.5525818, -14.4918137, 14.6026573
34: -29.7145615, -10.5731306, -29.7182007, -10.5902529, -10.6581154, 10.6651726
35: -29.2096062, -9.9421520, -29.2155190, -9.9631252, -10.6853294, 10.6953697
36: -31.8682117, -9.4012384, -31.8624802, -9.4165535, -12.6916084, 12.6771393
37: -46.1319923, -23.4822350, -46.1427231, -23.4925327, -16.0659409, 16.0703583
38: -34.2868538, -11.5196295, -34.2858849, -11.5259581, -15.1198654, 15.0871353
39: -56.3104095, -30.6322899, -56.3128204, -30.6573143, -13.1453819, 13.2211609
40: -40.2447205, -23.2933044, -40.2447357, -23.2888680, -8.1909142, 8.1856785
41: -26.7289772, -6.9915156, -26.7242775, -6.9746871, -11.2670174, 11.2222748
42: -14.5358238, -2.0983176, -14.5412750, -2.1184959, -8.5102119, 8.5273457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0822612, upper bound: 5.0837803
time: 17.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0822661, upper bound: 5.0911285
time: 17.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.2120132, 0.6894102, -20.2457123, 0.7570481, -13.4646187, 13.4476318
1: -6.4214463, 5.2770166, -6.4473600, 5.3228936, -6.4232025, 6.4225731
2: -11.0153522, 2.2172618, -11.0229931, 2.2723191, -8.5906372, 8.5583210
3: -12.3319073, 3.4154320, -12.3455276, 3.4483309, -11.1789398, 11.1711426
4: -22.0437851, -5.6449938, -22.0533981, -5.5828009, -9.2947693, 9.2587738
5: -10.7907591, 5.6406746, -10.8032789, 5.7115431, -12.2156372, 12.1633911
6: -22.4883022, -4.4998531, -22.4955177, -4.4321003, -11.2349968, 11.1565323
7: -9.5470362, 8.9116154, -9.5691395, 8.9635906, -12.4754181, 12.4893761
8: -26.3444901, -5.6134253, -26.3686924, -5.5550494, -9.8424110, 9.8393173
9: -14.5812054, 2.1821284, -14.6036129, 2.2007890, -12.9464493, 12.9849434
10: -5.8905816, 11.7451620, -5.9202662, 11.7595434, -13.2427864, 13.2766609
11: 9.5957050, 21.1072502, 9.5220528, 21.1119919, -7.4650383, 7.5329399
12: -15.1524525, 9.8482332, -15.1856308, 9.8812265, -18.5336456, 18.4446640
13: -28.0259666, -3.0119934, -28.0402393, -3.0086377, -12.9129715, 12.9508057
14: -31.3886986, 0.6476603, -31.4106102, 0.6737332, -21.5249939, 21.5264664
15: -24.9073792, -10.7093525, -24.9354630, -10.6822605, -8.8750153, 8.8922539
16: -6.9421759, 7.9062872, -6.9569163, 7.9385176, -10.1457596, 10.1438522
17: -14.7806721, 11.7459602, -14.7706108, 11.7779503, -21.7625046, 21.7089157
18: -0.9312122, 12.5703230, -0.9302588, 12.5818768, -10.7398643, 10.6998482
19: -5.2992721, 4.7391400, -5.3228397, 4.7410903, -7.5736599, 7.5990372
20: -3.3992128, 7.9683557, -3.4371712, 7.9875593, -10.2532616, 10.3000984
21: -1.9460015, 8.8795700, -1.9805188, 8.8838310, -8.9107018, 8.9650803
22: -9.2154169, 2.7633781, -9.2405090, 2.7836452, -8.6838570, 8.6901531
23: 1.3619481, 12.5044355, 1.2958186, 12.5064163, -7.7139130, 7.7830429
24: -2.6833773, 10.4953737, -2.7492561, 10.4964333, -8.1500587, 8.2260742
25: 0.3855484, 13.7563047, 0.2789836, 13.7624054, -9.3493004, 9.4682732
26: -17.4163132, 2.4737883, -17.4473343, 2.4868279, -14.5960083, 14.6106567
27: -10.3191471, 6.2871437, -10.3131762, 6.2955747, -9.1643772, 9.1488819
28: 1.0805938, 13.5793371, 1.0336273, 13.5794640, -9.6228180, 9.6802750
29: -5.1139002, 8.3466358, -5.1282759, 8.3587227, -8.6282597, 8.6290054
30: 5.9918137, 17.7092743, 5.9213462, 17.7146225, -7.6345081, 7.7250366
31: -3.3822317, 10.3893347, -3.4191146, 10.3914566, -9.1518135, 9.1977196
32: -19.5920849, -2.7449851, -19.6010780, -2.6973252, -10.7633514, 10.7090492
33: -47.0181236, -21.5340195, -47.0388260, -21.5363693, -14.5163193, 14.5792084
34: -29.7158928, -10.5797234, -29.7327576, -10.5781021, -10.6769257, 10.6707191
35: -29.2096672, -9.9547272, -29.2298622, -9.9428062, -10.7255363, 10.6838379
36: -31.8857193, -9.4049063, -31.9007835, -9.3792448, -12.7556992, 12.6958733
37: -46.1175995, -23.4944992, -46.1323853, -23.4924316, -16.0759583, 16.0395889
38: -34.3040619, -11.5249710, -34.3209229, -11.4948444, -15.1687851, 15.1070251
39: -56.3039398, -30.6440372, -56.3192558, -30.6401558, -13.1734810, 13.2114639
40: -40.2390594, -23.2979851, -40.2495193, -23.2783470, -8.2040482, 8.1822205
41: -26.7415924, -6.9933128, -26.7554893, -6.9461293, -11.3085823, 11.2434273
42: -14.5322437, -2.1001043, -14.5452690, -2.1121945, -8.5158024, 8.5264053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0786667
time: 24.23 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0859254
time: 25.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -20.2128677, 0.7067225, -20.2261467, 0.6996210, -13.4045258, 13.4405746
1: -6.4214144, 5.2840853, -6.4358296, 5.2869692, -6.3883762, 6.4127903
2: -11.0199566, 2.2297025, -11.0155449, 2.2490616, -8.5731277, 8.5516071
3: -12.3312979, 3.4241824, -12.3307648, 3.4091311, -11.1472435, 11.1607590
4: -22.0400887, -5.6379204, -22.0388260, -5.6269302, -9.2406693, 9.2519989
5: -10.7945080, 5.6562004, -10.7910671, 5.6765280, -12.1883011, 12.1560135
6: -22.4924583, -4.4988785, -22.4883366, -4.4320393, -11.2393799, 11.1480713
7: -9.5554457, 8.9276133, -9.5531940, 8.9298830, -12.4585724, 12.4719772
8: -26.3461533, -5.6042242, -26.3504906, -5.5972357, -9.8103600, 9.8193054
9: -14.5785589, 2.1873331, -14.5893059, 2.1421764, -12.8829422, 12.9875641
10: -5.8938103, 11.7494392, -5.9169660, 11.7242279, -13.2066231, 13.2790642
11: 9.5744104, 21.1080837, 9.5534878, 21.1059952, -7.4829664, 7.4972439
12: -15.1651754, 9.8545275, -15.1676607, 9.8762484, -18.5260315, 18.4525757
13: -28.0189285, -3.0104206, -28.0253944, -3.0650682, -12.8547058, 12.9423027
14: -31.3986492, 0.6610794, -31.4073410, 0.6641459, -21.5229187, 21.5271072
15: -24.9215927, -10.6884594, -24.9441872, -10.6943884, -8.8557167, 8.9151325
16: -6.9408989, 7.9069681, -6.9438238, 7.8907747, -10.0974007, 10.1355858
17: -14.7854481, 11.7588463, -14.7608566, 11.7668772, -21.7572632, 21.6985703
18: -0.9305103, 12.5663023, -0.8874967, 12.5691738, -10.7357788, 10.6565475
19: -5.3020811, 4.7390471, -5.2885170, 4.7358809, -7.5753288, 7.5655785
20: -3.4051933, 7.9678040, -3.4041114, 7.9826803, -10.2584381, 10.2701530
21: -1.9485534, 8.8772221, -1.9465891, 8.8789215, -8.9136238, 8.9289665
22: -9.2262135, 2.7787385, -9.2113571, 2.7738662, -8.6729355, 8.6733589
23: 1.3562821, 12.5036488, 1.3398991, 12.4976521, -7.7152977, 7.7349014
24: -2.6835995, 10.4927073, -2.7098172, 10.4870300, -8.1448250, 8.1824970
25: 0.3818998, 13.7609196, 0.3099346, 13.7558060, -9.3446617, 9.4361954
26: -17.4268093, 2.4740682, -17.3937702, 2.4736514, -14.5890656, 14.5570221
27: -10.3198366, 6.2817278, -10.2646484, 6.2822971, -9.1616325, 9.0920639
28: 1.0776448, 13.5751381, 1.0825281, 13.5692616, -9.6275787, 9.6276703
29: -5.1189914, 8.3508301, -5.0912952, 8.3466206, -8.6188812, 8.5903206
30: 5.9857244, 17.7101479, 5.9461832, 17.7091484, -7.6371326, 7.6975651
31: -3.3858588, 10.3908567, -3.3794231, 10.3858709, -9.1547699, 9.1584148
32: -19.5956345, -2.7438262, -19.5922756, -2.6973238, -10.7630234, 10.6973877
33: -47.0342102, -21.5148716, -47.0404358, -21.5547810, -14.5044594, 14.5944176
34: -29.7242088, -10.5722342, -29.7255764, -10.5781927, -10.6780548, 10.6650162
35: -29.2211227, -9.9408255, -29.2259331, -9.9442968, -10.7153320, 10.6950150
36: -31.8917503, -9.4004192, -31.8852005, -9.3808975, -12.7502480, 12.6818619
37: -46.1335793, -23.4826698, -46.1376686, -23.4930172, -16.0693512, 16.0651016
38: -34.3057938, -11.5181990, -34.3053169, -11.4909525, -15.1736679, 15.0916710
39: -56.3152504, -30.6319218, -56.3179855, -30.6692810, -13.1483002, 13.2205811
40: -40.2493591, -23.2933426, -40.2501640, -23.2867889, -8.1945572, 8.1882782
41: -26.7475300, -6.9905887, -26.7424030, -6.9482489, -11.3097477, 11.2317162
42: -14.5378981, -2.0980291, -14.5398426, -2.1155839, -8.5164242, 8.5223732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0816436, upper bound: 5.0837803
time: 14.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0816475, upper bound: 5.0911285
time: 7.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20.2220516, 0.7072961, -20.2461300, 0.7662809, -13.4851570, 13.4513359
1: -6.4267154, 5.2842469, -6.4476337, 5.3270359, -6.4373589, 6.4189224
2: -11.0233603, 2.2299030, -11.0232582, 2.2798021, -8.6097641, 8.5558758
3: -12.3381319, 3.4241695, -12.3459511, 3.4536059, -11.1966400, 11.1681671
4: -22.0469284, -5.6378031, -22.0533733, -5.5788870, -9.2998276, 9.2594833
5: -10.8003597, 5.6563253, -10.8035536, 5.7206459, -12.2377930, 12.1619034
6: -22.4932213, -4.5000052, -22.4971294, -4.4321775, -11.2423897, 11.1574669
7: -9.5623312, 8.9276180, -9.5703983, 8.9731617, -12.5118027, 12.4833107
8: -26.3543854, -5.6041775, -26.3693695, -5.5494151, -9.8705788, 9.8296852
9: -14.5861683, 2.1874046, -14.6065874, 2.2019198, -12.9464722, 12.9955063
10: -5.8947053, 11.7496529, -5.9224100, 11.7606735, -13.2467804, 13.2841835
11: 9.5731630, 21.1120834, 9.5199242, 21.1148796, -7.4895535, 7.5309505
12: -15.1652851, 9.8552799, -15.1927528, 9.8816156, -18.5314026, 18.4693375
13: -28.0256672, -3.0097156, -28.0400009, -3.0072000, -12.9214478, 12.9501915
14: -31.3975410, 0.6614695, -31.4114838, 0.6819391, -21.5419312, 21.5376205
15: -24.9210320, -10.6873512, -24.9436455, -10.6810169, -8.8714104, 8.9166279
16: -6.9467859, 7.9070892, -6.9577312, 7.9387040, -10.1540489, 10.1446762
17: -14.7879467, 11.7590408, -14.7708206, 11.7851954, -21.7716980, 21.7134933
18: -0.9304857, 12.5720272, -0.9308510, 12.5823298, -10.7423363, 10.7019844
19: -5.3024693, 4.7419767, -5.3236370, 4.7422671, -7.5789490, 7.6023560
20: -3.4059377, 7.9703484, -3.4385695, 7.9887166, -10.2642059, 10.3040466
21: -1.9492741, 8.8795090, -1.9815418, 8.8838873, -8.9174957, 8.9649677
22: -9.2264805, 2.7831149, -9.2466488, 2.7845230, -8.6804008, 8.7138481
23: 1.3559707, 12.5086632, 1.2950292, 12.5084133, -7.7211132, 7.7855930
24: -2.6838713, 10.4969683, -2.7496896, 10.4968910, -8.1509342, 8.2287598
25: 0.3816042, 13.7637091, 0.2769675, 13.7629910, -9.3504581, 9.4751701
26: -17.4271660, 2.4802926, -17.4532909, 2.4871464, -14.5960159, 14.6254120
27: -10.3202438, 6.2879686, -10.3141279, 6.2959080, -9.1689796, 9.1502705
28: 1.0772958, 13.5800028, 1.0323205, 13.5795603, -9.6331863, 9.6809311
29: -5.1191969, 8.3561392, -5.1313691, 8.3591995, -8.6271362, 8.6386681
30: 5.9850760, 17.7126961, 5.9187746, 17.7147255, -7.6418381, 7.7260628
31: -3.3863435, 10.3945312, -3.4196863, 10.3938665, -9.1595421, 9.2010860
32: -19.5963459, -2.7446969, -19.6020927, -2.6971786, -10.7673759, 10.7111454
33: -47.0350723, -21.5141716, -47.0484085, -21.5350800, -14.5145493, 14.6115570
34: -29.7246399, -10.5729866, -29.7380257, -10.5776758, -10.6810493, 10.6814880
35: -29.2212772, -9.9417734, -29.2367420, -9.9423122, -10.7193832, 10.7093620
36: -31.8918076, -9.4009571, -31.9043961, -9.3789778, -12.7534409, 12.7057381
37: -46.1334114, -23.4838219, -46.1418076, -23.4917831, -16.0710754, 16.0730057
38: -34.3059158, -11.5194702, -34.3212700, -11.4917889, -15.1739273, 15.1096382
39: -56.3193855, -30.6316452, -56.3290329, -30.6391029, -13.1730995, 13.2304955
40: -40.2500992, -23.2932758, -40.2549210, -23.2783604, -8.2064705, 8.1931305
41: -26.7479591, -6.9913149, -26.7584324, -6.9458885, -11.3153172, 11.2488518
42: -14.5386248, -2.0980027, -14.5476608, -2.1119525, -8.5206585, 8.5328083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 641

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0837803
time: 22.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0911285
time: 16.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 41.75 seconds
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0626865, upper bound: 5.0837445
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0626865, upper bound: 5.0910806
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0721745, upper bound: 5.0837445
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0721745, upper bound: 5.0910806
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0715371, upper bound: 5.0837445
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0626865, upper bound: 5.0910806
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0810243, upper bound: 5.0837445
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0810243, upper bound: 5.0910806
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0745832
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0818185
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0797499
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0870523
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0727759, upper bound: 5.0837803
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0727807, upper bound: 5.0911285
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0822612, upper bound: 5.0837803
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0822661, upper bound: 5.0911285
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0786667
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0859254
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0816436, upper bound: 5.0837803
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0816475, upper bound: 5.0911285
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0837803
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.75
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0911285

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.2148170, 0.7073524, -20.1968613, 0.6771173, -13.3894691, 13.3722992
1: -6.4212217, 5.2864671, -6.4082870, 5.2640505, -6.3671646, 6.3687401
2: -11.0113993, 2.2426033, -10.9949818, 2.2164993, -8.5303650, 8.5501518
3: -12.3300095, 3.4390161, -12.3165455, 3.3897500, -11.1239548, 11.1630936
4: -22.0347061, -5.6171932, -22.0223522, -5.6508093, -9.2070160, 9.2563095
5: -10.7851648, 5.6716714, -10.7685957, 5.6347313, -12.1345673, 12.1556931
6: -22.4596024, -4.4849362, -22.4218445, -4.5179319, -11.1189613, 11.1135674
7: -9.5560188, 8.9352760, -9.5414886, 8.9142952, -12.4426956, 12.4654388
8: -26.3461590, -5.5971937, -26.3213959, -5.6199365, -9.7933807, 9.7856560
9: -14.5789299, 2.1918094, -14.5655947, 2.1207271, -12.8637466, 12.9575386
10: -5.8968234, 11.7580214, -5.8864179, 11.6917458, -13.1811752, 13.2471924
11: 9.5585270, 21.0901260, 9.6067133, 21.0712547, -7.4763212, 7.4207096
12: -15.1632442, 9.8562412, -15.1425791, 9.8497248, -18.4884033, 18.4307938
13: -28.0256310, -3.0118914, -28.0026588, -3.0931456, -12.8394928, 12.9146080
14: -31.4176235, 0.6467314, -31.3452396, 0.6313486, -21.5145416, 21.4326401
15: -24.9227581, -10.6910458, -24.9238834, -10.7139339, -8.8402710, 8.8820591
16: -6.9417305, 7.9152155, -6.9285288, 7.8756828, -10.0860329, 10.1111393
17: -14.8010921, 11.7553196, -14.7267237, 11.7559862, -21.7578888, 21.6575928
18: -0.9361885, 12.5666409, -0.8676720, 12.5569019, -10.7275009, 10.6336136
19: -5.3107190, 4.7378883, -5.2684112, 4.7316847, -7.5809517, 7.5405998
20: -3.4094818, 7.9705939, -3.3858609, 7.9589381, -10.2320061, 10.2558746
21: -1.9529928, 8.8797007, -1.9219947, 8.8676338, -8.9098282, 8.9088612
22: -9.2337408, 2.7778754, -9.1961908, 2.7678022, -8.6713905, 8.6557064
23: 1.3355381, 12.4927120, 1.3861718, 12.4773293, -7.7245483, 7.6741905
24: -2.7005455, 10.4728870, -2.6451321, 10.4475870, -8.1350021, 8.0896664
25: 0.3544080, 13.7316475, 0.4031770, 13.7007904, -9.3338394, 9.3082542
26: -17.4383717, 2.4762855, -17.3727989, 2.4664037, -14.5948868, 14.5337524
27: -10.3257856, 6.2808757, -10.2530708, 6.2704983, -9.1433220, 9.0711765
28: 1.0591691, 13.5671148, 1.1209195, 13.5537834, -9.6365509, 9.5779495
29: -5.1303172, 8.3483429, -5.0749197, 8.3364305, -8.6194286, 8.5682449
30: 5.9737782, 17.6898975, 6.0057259, 17.6693153, -7.6211586, 7.6140156
31: -3.3989058, 10.3860941, -3.3528175, 10.3765059, -9.1582336, 9.1245632
32: -19.5634155, -2.7338994, -19.5236835, -2.7795506, -10.6410751, 10.6590061
33: -47.0286865, -21.5051765, -47.0185318, -21.5832195, -14.4687614, 14.5878639
34: -29.7132149, -10.5635252, -29.6966858, -10.5987387, -10.6379700, 10.6535454
35: -29.2106686, -9.9362755, -29.1976757, -9.9719782, -10.6693077, 10.6802597
36: -31.8789330, -9.3994923, -31.8395691, -9.4300098, -12.6740570, 12.6485710
37: -46.1384544, -23.4785004, -46.1269531, -23.4975624, -16.0632172, 16.0378189
38: -34.2908783, -11.5147781, -34.2608261, -11.5324173, -15.1134567, 15.0633850
39: -56.3074379, -30.6287212, -56.2933846, -30.6922894, -13.1103745, 13.2115326
40: -40.2397842, -23.2892952, -40.2270622, -23.3104477, -8.1606026, 8.1750240
41: -26.7225075, -6.9792619, -26.6891918, -7.0093670, -11.2225266, 11.2010765
42: -14.5389252, -2.0966508, -14.5247068, -2.1272411, -8.5040779, 8.5109577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0574838, upper bound: 5.0908049
time: 19.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0624141, upper bound: 5.0908049
time: 47.23 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2240009, 0.7079577, -20.2168636, 0.7437630, -13.4701233, 13.3830872
1: -6.4265375, 5.2866001, -6.4200811, 5.3041167, -6.4161682, 6.3748703
2: -11.0148029, 2.2427757, -11.0027399, 2.2471962, -8.5669861, 8.5544224
3: -12.3368196, 3.4389825, -12.3317327, 3.4342630, -11.1733398, 11.1705170
4: -22.0415382, -5.6170788, -22.0368843, -5.6027975, -9.2661781, 9.2638054
5: -10.7910271, 5.6718216, -10.7811127, 5.6788721, -12.1840515, 12.1615944
6: -22.4603481, -4.4860439, -22.4306126, -4.5181007, -11.1219521, 11.1229553
7: -9.5628653, 8.9352589, -9.5586605, 8.9575443, -12.4959412, 12.4767609
8: -26.3543549, -5.5971465, -26.3402786, -5.5721250, -9.8535919, 9.7960358
9: -14.5865231, 2.1918674, -14.5828934, 2.1804612, -12.9272461, 12.9654846
10: -5.8977404, 11.7582607, -5.8918371, 11.7281818, -13.2213440, 13.2523079
11: 9.5572624, 21.0941048, 9.5731287, 21.0801601, -7.4829121, 7.4544258
12: -15.1633348, 9.8570385, -15.1676311, 9.8550720, -18.4937286, 18.4475174
13: -28.0324936, -3.0111718, -28.0172443, -3.0352209, -12.9062347, 12.9225311
14: -31.4166145, 0.6470780, -31.3494492, 0.6492119, -21.5335464, 21.4430847
15: -24.9221954, -10.6899481, -24.9233265, -10.7005711, -8.8559875, 8.8835583
16: -6.9476295, 7.9153552, -6.9424362, 7.9236269, -10.1426888, 10.1202354
17: -14.8035831, 11.7555141, -14.7366753, 11.7742872, -21.7723007, 21.6724777
18: -0.9361782, 12.5723686, -0.9110301, 12.5700569, -10.7340546, 10.6790504
19: -5.3110943, 4.7408118, -5.3035269, 4.7380667, -7.5845661, 7.5773716
20: -3.4102125, 7.9731312, -3.4203444, 7.9649549, -10.2377663, 10.2897568
21: -1.9537058, 8.8820105, -1.9569391, 8.8726120, -8.9137077, 8.9448624
22: -9.2340193, 2.7822466, -9.2314692, 2.7784646, -8.6788578, 8.6961918
23: 1.3352342, 12.4977703, 1.3413234, 12.4880524, -7.7303524, 7.7248783
24: -2.7008371, 10.4771843, -2.6849463, 10.4574909, -8.1411343, 8.1359215
25: 0.3540947, 13.7344208, 0.3702252, 13.7079754, -9.3396378, 9.3472137
26: -17.4387512, 2.4824803, -17.4322777, 2.4798985, -14.6018066, 14.6020966
27: -10.3261871, 6.2871151, -10.3025265, 6.2841082, -9.1506615, 9.1293869
28: 1.0588019, 13.5719643, 1.0707371, 13.5640602, -9.6421776, 9.6312103
29: -5.1305094, 8.3536568, -5.1150007, 8.3490267, -8.6277008, 8.6165619
30: 5.9731140, 17.6924286, 5.9783292, 17.6748772, -7.6258583, 7.6425228
31: -3.3994071, 10.3897657, -3.3930504, 10.3845081, -9.1630211, 9.1672192
32: -19.5641327, -2.7347772, -19.5335350, -2.7793801, -10.6454124, 10.6727600
33: -47.0294952, -21.5045013, -47.0265350, -21.5635815, -14.4788513, 14.6050072
34: -29.7136478, -10.5643091, -29.7090836, -10.5982256, -10.6409645, 10.6700287
35: -29.2108097, -9.9373074, -29.2084789, -9.9699888, -10.6733627, 10.6946068
36: -31.8790092, -9.4000349, -31.8587151, -9.4281635, -12.6772499, 12.6724319
37: -46.1382751, -23.4796753, -46.1310349, -23.4963036, -16.0649948, 16.0457077
38: -34.2909775, -11.5160780, -34.2768402, -11.5332222, -15.1137314, 15.0813599
39: -56.3115082, -30.6283989, -56.3043861, -30.6620140, -13.1351852, 13.2214584
40: -40.2405281, -23.2892818, -40.2317772, -23.3020229, -8.1725197, 8.1798725
41: -26.7229137, -6.9799623, -26.7052994, -7.0070024, -11.2280960, 11.2182159
42: -14.5396471, -2.0965858, -14.5325289, -2.1236348, -8.5083008, 8.5213947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0669220, upper bound: 5.0908049
time: 16.21 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0719021, upper bound: 5.0908049
time: 18.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.2152157, 0.7165787, -20.2168865, 0.6941495, -13.4006729, 13.4041367
1: -6.4213591, 5.2960577, -6.4284024, 5.2818079, -6.3795567, 6.3984852
2: -11.0165644, 2.2442479, -11.0045185, 2.2272542, -8.5447006, 8.5586262
3: -12.3311539, 3.4394135, -12.3215599, 3.3940823, -11.1280289, 11.1685524
4: -22.0355854, -5.6152191, -22.0289307, -5.6461301, -9.2109718, 9.2689705
5: -10.7898788, 5.6725130, -10.7781258, 5.6495891, -12.1536789, 12.1633377
6: -22.4819603, -4.4845581, -22.4611607, -4.4766335, -11.1823311, 11.1410828
7: -9.5559034, 8.9384022, -9.5473347, 8.9215565, -12.4488068, 12.4747658
8: -26.3465614, -5.5878706, -26.3429050, -5.6032219, -9.8050957, 9.8163338
9: -14.5800352, 2.1979182, -14.5815830, 2.1338022, -12.8756256, 12.9862518
10: -5.8976054, 11.7683067, -5.9072680, 11.7125912, -13.1981964, 13.2854118
11: 9.5579529, 21.0995522, 9.5879221, 21.0879440, -7.4875393, 7.4496784
12: -15.1699162, 9.8562956, -15.1571474, 9.8672962, -18.5128708, 18.4414673
13: -28.0297947, -3.0101762, -28.0108376, -3.0779524, -12.8530159, 12.9249611
14: -31.4184532, 0.6587734, -31.3741684, 0.6533744, -21.5318527, 21.4769440
15: -24.9233627, -10.6851921, -24.9386215, -10.7026548, -8.8493538, 8.9089165
16: -6.9422107, 7.9193325, -6.9396467, 7.8841968, -10.0916443, 10.1272125
17: -14.8022442, 11.7561893, -14.7353048, 11.7587652, -21.7641373, 21.6658401
18: -0.9371386, 12.5678072, -0.8753879, 12.5616550, -10.7379494, 10.6436005
19: -5.3118367, 4.7384024, -5.2730784, 4.7320800, -7.5829926, 7.5462246
20: -3.4123406, 7.9715052, -3.3933449, 7.9691515, -10.2457771, 10.2622490
21: -1.9539979, 8.8820400, -1.9302406, 8.8722477, -8.9140244, 8.9177742
22: -9.2362700, 2.7785175, -9.2032681, 2.7695537, -8.6763344, 8.6635475
23: 1.3348775, 12.4972544, 1.3726773, 12.4850311, -7.7285042, 7.6923409
24: -2.7011435, 10.4856024, -2.6710701, 10.4698277, -8.1522579, 8.1289291
25: 0.3539901, 13.7473831, 0.3704083, 13.7278366, -9.3531132, 9.3565941
26: -17.4402046, 2.4752178, -17.3772392, 2.4665101, -14.5973053, 14.5389709
27: -10.3278236, 6.2820148, -10.2606506, 6.2724957, -9.1575775, 9.0867767
28: 1.0571694, 13.5703449, 1.1116538, 13.5593300, -9.6416016, 9.5912857
29: -5.1330643, 8.3504553, -5.0823035, 8.3410902, -8.6280861, 8.5785675
30: 5.9735012, 17.7035923, 5.9781809, 17.6932926, -7.6369152, 7.6555729
31: -3.3989680, 10.3875284, -3.3555977, 10.3759947, -9.1602898, 9.1300201
32: -19.5876217, -2.7335174, -19.5670624, -2.7361324, -10.7089348, 10.6905174
33: -47.0355072, -21.5036678, -47.0312271, -21.5658112, -14.4914970, 14.5967636
34: -29.7233200, -10.5633488, -29.7163811, -10.5861340, -10.6608887, 10.6697998
35: -29.2223244, -9.9359570, -29.2189007, -9.9511738, -10.7033539, 10.6944618
36: -31.9026299, -9.3991642, -31.8814392, -9.3924770, -12.7358475, 12.6771393
37: -46.1399002, -23.4801788, -46.1262169, -23.4968319, -16.0683746, 16.0407333
38: -34.3097839, -11.5146790, -34.2962036, -11.4983006, -15.1675110, 15.0859222
39: -56.3164062, -30.6280518, -56.3097305, -30.6740494, -13.1380730, 13.2208023
40: -40.2451706, -23.2892380, -40.2371941, -23.2999325, -8.1761551, 8.1824303
41: -26.7414780, -6.9790459, -26.7232895, -6.9805593, -11.2708111, 11.2276344
42: -14.5417118, -2.0963197, -14.5310726, -2.1207418, -8.5144939, 8.5164337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0663335, upper bound: 5.0908049
time: 8.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0712640, upper bound: 5.0908049
time: 26.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2244110, 0.7171698, -20.2368565, 0.7607715, -13.4813232, 13.4149284
1: -6.4266663, 5.2962046, -6.4401999, 5.3218498, -6.4285469, 6.4045963
2: -11.0199909, 2.2444203, -11.0122709, 2.2579439, -8.5813255, 8.5628929
3: -12.3379631, 3.4393814, -12.3367481, 3.4386406, -11.1774330, 11.1759720
4: -22.0424080, -5.6150632, -22.0434532, -5.5980911, -9.2701416, 9.2764549
5: -10.7957411, 5.6726542, -10.7906151, 5.6936936, -12.2031708, 12.1692314
6: -22.4827614, -4.4856954, -22.4699497, -4.4767828, -11.1853256, 11.1504898
7: -9.5627613, 8.9384365, -9.5645809, 8.9648113, -12.5020370, 12.4860878
8: -26.3547592, -5.5877910, -26.3618031, -5.5554008, -9.8653030, 9.8267288
9: -14.5876198, 2.1979671, -14.5988722, 2.1935191, -12.9391327, 12.9942093
10: -5.8984957, 11.7685452, -5.9127159, 11.7490101, -13.2383575, 13.2905617
11: 9.5567093, 21.1035500, 9.5543957, 21.0968285, -7.4941263, 7.4833984
12: -15.1700335, 9.8571081, -15.1822205, 9.8726454, -18.5182190, 18.4581985
13: -28.0366535, -3.0094547, -28.0254784, -3.0200555, -12.9197464, 12.9328690
14: -31.4173737, 0.6592405, -31.3782978, 0.6712294, -21.5509338, 21.4874420
15: -24.9228096, -10.6840954, -24.9380951, -10.6892853, -8.8650703, 8.9104099
16: -6.9481111, 7.9194698, -6.9535327, 7.9320951, -10.1482849, 10.1362839
17: -14.8047609, 11.7564135, -14.7452717, 11.7770844, -21.7785797, 21.6807251
18: -0.9370947, 12.5735283, -0.9187429, 12.5747929, -10.7445107, 10.6890259
19: -5.3122168, 4.7413368, -5.3081732, 4.7384715, -7.5866165, 7.5829887
20: -3.4130828, 7.9740529, -3.4278028, 7.9751658, -10.2515335, 10.2961349
21: -1.9547237, 8.8843374, -1.9651606, 8.8771906, -8.9179077, 8.9537697
22: -9.2365561, 2.7829063, -9.2385616, 2.7802382, -8.6837769, 8.7040367
23: 1.3345609, 12.5022678, 1.3278346, 12.4957495, -7.7343102, 7.7430382
24: -2.7014112, 10.4898844, -2.7109258, 10.4796963, -8.1583824, 8.1751900
25: 0.3536658, 13.7501812, 0.3374615, 13.7350368, -9.3588982, 9.3955612
26: -17.4406128, 2.4814205, -17.4367313, 2.4800239, -14.6042328, 14.6073303
27: -10.3282499, 6.2882385, -10.3101521, 6.2860818, -9.1649055, 9.1449776
28: 1.0568202, 13.5752020, 1.0614882, 13.5696144, -9.6472092, 9.6445503
29: -5.1332779, 8.3557491, -5.1223774, 8.3536720, -8.6363373, 8.6269073
30: 5.9728584, 17.7061424, 5.9507952, 17.6988487, -7.6416111, 7.6840687
31: -3.3994677, 10.3911934, -3.3958211, 10.3839912, -9.1650543, 9.1726704
32: -19.5883255, -2.7344007, -19.5769138, -2.7359867, -10.7132874, 10.7042713
33: -47.0363121, -21.5029621, -47.0391769, -21.5461121, -14.5015831, 14.6138802
34: -29.7237225, -10.5641317, -29.7288284, -10.5856552, -10.6638794, 10.6862831
35: -29.2224884, -9.9369259, -29.2296944, -9.9491940, -10.7073898, 10.7087822
36: -31.9026394, -9.3997374, -31.9006653, -9.3906345, -12.7390480, 12.7010117
37: -46.1397362, -23.4813042, -46.1303482, -23.4955254, -16.0701065, 16.0486298
38: -34.3099251, -11.5159492, -34.3122330, -11.4990578, -15.1678085, 15.1038589
39: -56.3205185, -30.6277275, -56.3208313, -30.6437759, -13.1628799, 13.2307396
40: -40.2459106, -23.2892399, -40.2419395, -23.2915096, -8.1880646, 8.1872654
41: -26.7419071, -6.9797535, -26.7393856, -6.9782147, -11.2763710, 11.2447662
42: -14.5424337, -2.0962472, -14.5389109, -2.1170902, -8.5187321, 8.5268650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0757550, upper bound: 5.0908049
time: 21.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0807517, upper bound: 5.0908049
time: 22.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -20.1968117, 0.6725457, -20.2376060, 0.7554336, -13.4557114, 13.4200134
1: -6.4145765, 5.2703252, -6.4439526, 5.3222923, -6.4192772, 6.4129715
2: -10.9998875, 2.2037613, -11.0141754, 2.2712989, -8.5775986, 8.5371990
3: -12.3073502, 3.3970208, -12.3318605, 3.4470665, -11.1593285, 11.1395760
4: -22.0171547, -5.6606116, -22.0383854, -5.5841141, -9.2715111, 9.2280388
5: -10.7713299, 5.6252851, -10.7924480, 5.7103615, -12.1949692, 12.1366196
6: -22.4680538, -4.5012293, -22.4876289, -4.4313307, -11.2050095, 11.1440620
7: -9.5273924, 8.8984623, -9.5589075, 8.9626293, -12.4559708, 12.4666443
8: -26.3170815, -5.6267972, -26.3532295, -5.5555143, -9.8251648, 9.8123169
9: -14.5632010, 2.1643147, -14.5942068, 2.1983542, -12.9335938, 12.9579926
10: -5.8774018, 11.7292213, -5.9145107, 11.7565231, -13.2293091, 13.2527771
11: 9.6148319, 21.0886803, 9.5250130, 21.1014481, -7.4382029, 7.5197144
12: -15.1229286, 9.8223057, -15.1834364, 9.8667955, -18.4878006, 18.4177933
13: -28.0195732, -3.0247226, -28.0392780, -3.0150969, -12.9014664, 12.9321442
14: -31.3597984, 0.6309817, -31.4057350, 0.6646109, -21.4838562, 21.5120773
15: -24.9096718, -10.7142086, -24.9375973, -10.6848688, -8.8728638, 8.8692760
16: -6.9310017, 7.8954034, -6.9513621, 7.9378228, -10.1396599, 10.1282806
17: -14.7522736, 11.7143850, -14.7686729, 11.7606087, -21.7170181, 21.6816864
18: -0.9098988, 12.5571938, -0.9274092, 12.5749674, -10.7076187, 10.6897354
19: -5.2923465, 4.7314196, -5.3210397, 4.7392077, -7.5657578, 7.5875130
20: -3.3880234, 7.9599190, -3.4337132, 7.9866691, -10.2407303, 10.2849121
21: -1.9367478, 8.8738155, -1.9777374, 8.8832407, -8.9014511, 8.9520874
22: -9.2067509, 2.7573800, -9.2394981, 2.7794313, -8.6712646, 8.6771049
23: 1.3718176, 12.4910898, 1.2975271, 12.5000362, -7.6982193, 7.7686481
24: -2.6761405, 10.4822454, -2.7481139, 10.4896450, -8.1360607, 8.2114487
25: 0.3965392, 13.7356033, 0.2806087, 13.7519293, -9.3280411, 9.4465103
26: -17.3933525, 2.4617133, -17.4450321, 2.4802508, -14.5641174, 14.6005020
27: -10.3097563, 6.2821426, -10.3105183, 6.2935109, -9.1496201, 9.1422691
28: 1.0924311, 13.5682964, 1.0358918, 13.5749035, -9.6064911, 9.6667747
29: -5.0997601, 8.3343925, -5.1271152, 8.3525038, -8.6093483, 8.6209221
30: 6.0081739, 17.6994400, 5.9230947, 17.7094116, -7.6226044, 7.7131252
31: -3.3735652, 10.3750782, -3.4160750, 10.3871498, -9.1392212, 9.1773663
32: -19.5777092, -2.7532647, -19.5950909, -2.6987519, -10.7460823, 10.6948357
33: -47.0024414, -21.5609016, -47.0308762, -21.5396843, -14.5138626, 14.5459671
34: -29.6969109, -10.5929575, -29.7230873, -10.5792580, -10.6717529, 10.6479340
35: -29.2069054, -9.9658833, -29.2295132, -9.9442253, -10.7181015, 10.6746902
36: -31.8758812, -9.4089680, -31.8988209, -9.3818531, -12.7383156, 12.6926689
37: -46.1068039, -23.5080433, -46.1297607, -23.4989147, -16.0397339, 16.0260315
38: -34.2912750, -11.5293722, -34.3174248, -11.4967766, -15.1417313, 15.1019821
39: -56.2884521, -30.6621895, -56.3109245, -30.6422787, -13.1715660, 13.1853905
40: -40.2280388, -23.3037624, -40.2437401, -23.2787495, -8.1803055, 8.1779060
41: -26.7287788, -7.0000095, -26.7502193, -6.9472194, -11.2857838, 11.2364388
42: -14.5292816, -2.1057103, -14.5429363, -2.1133637, -8.4921951, 8.5239601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0857781, upper bound: 5.0743069
time: 51.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0908483, upper bound: 5.0743069
time: 7.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.2058601, 0.6822584, -20.2402458, 0.7563171, -13.4735222, 13.4191818
1: -6.4190545, 5.2817278, -6.4456806, 5.3226271, -6.4242001, 6.4206409
2: -11.0086708, 2.2193568, -11.0183086, 2.2716775, -8.5836716, 8.5571556
3: -12.3199291, 3.4132910, -12.3374844, 3.4474399, -11.1696663, 11.1627617
4: -22.0305061, -5.6362734, -22.0458755, -5.5835171, -9.2800026, 9.2606659
5: -10.7827349, 5.6428537, -10.7981100, 5.7107611, -12.2030563, 12.1610298
6: -22.4818287, -4.4865198, -22.4938927, -4.4311881, -11.2172737, 11.1651421
7: -9.5384665, 8.9111576, -9.5639057, 8.9630508, -12.4637222, 12.4816170
8: -26.3323383, -5.6097207, -26.3605156, -5.5553112, -9.8346748, 9.8282623
9: -14.5714130, 2.1770875, -14.5966797, 2.1992624, -12.9424820, 12.9718437
10: -5.8904772, 11.7512636, -5.9188871, 11.7576618, -13.2415085, 13.2771492
11: 9.5937138, 21.1006870, 9.5232468, 21.1078339, -7.4679632, 7.5311375
12: -15.1310663, 9.8273163, -15.1843834, 9.8678160, -18.4943008, 18.4251328
13: -28.0322323, -3.0139122, -28.0397987, -3.0111394, -12.9180756, 12.9414940
14: -31.3837929, 0.6410775, -31.4073296, 0.6692491, -21.5186539, 21.5229797
15: -24.9124336, -10.7071877, -24.9375114, -10.6833096, -8.8778000, 8.8751106
16: -6.9406414, 7.9078355, -6.9551401, 7.9381037, -10.1481819, 10.1379681
17: -14.7717237, 11.7223377, -14.7696486, 11.7646961, -21.7421417, 21.6913605
18: -0.9185295, 12.5633564, -0.9281416, 12.5762691, -10.7214470, 10.6971512
19: -5.3041754, 4.7345462, -5.3217688, 4.7404032, -7.5811539, 7.5914726
20: -3.3975823, 7.9646540, -3.4349720, 7.9870420, -10.2503510, 10.2921371
21: -1.9455279, 8.8801355, -1.9789257, 8.8835335, -8.9127808, 8.9589844
22: -9.2183399, 2.7643523, -9.2399406, 2.7825928, -8.6854954, 8.6835098
23: 1.3486094, 12.5016212, 1.2968268, 12.5058994, -7.7279434, 7.7774410
24: -2.6948671, 10.4937735, -2.7484536, 10.4954453, -8.1625023, 8.2204781
25: 0.3667099, 13.7532368, 0.2800143, 13.7613344, -9.3678360, 9.4603424
26: -17.4094105, 2.4663320, -17.4458351, 2.4812036, -14.5827827, 14.6058426
27: -10.3189526, 6.2864881, -10.3112679, 6.2941613, -9.1600876, 9.1470451
28: 1.0693786, 13.5757847, 1.0349505, 13.5788622, -9.6345024, 9.6738014
29: -5.1155243, 8.3420210, -5.1276522, 8.3560867, -8.6288853, 8.6274490
30: 5.9932222, 17.7090530, 5.9221787, 17.7141533, -7.6416473, 7.7207317
31: -3.3892920, 10.3825321, -3.4169276, 10.3905659, -9.1589890, 9.1844120
32: -19.5893936, -2.7417319, -19.5994835, -2.6982911, -10.7575989, 10.7143402
33: -47.0087318, -21.5459137, -47.0323906, -21.5381985, -14.5204659, 14.5662308
34: -29.7046471, -10.5826283, -29.7258472, -10.5787430, -10.6785126, 10.6659164
35: -29.2114487, -9.9594498, -29.2302971, -9.9436703, -10.7227058, 10.6845627
36: -31.8882122, -9.4035778, -31.8996296, -9.3805561, -12.7477188, 12.7003479
37: -46.1165924, -23.4999161, -46.1307755, -23.4960480, -16.0541763, 16.0351105
38: -34.2995338, -11.5228186, -34.3187027, -11.4961052, -15.1502075, 15.1095352
39: -56.2924347, -30.6560745, -56.3117142, -30.6413174, -13.1745300, 13.1990128
40: -40.2367172, -23.2993813, -40.2476997, -23.2786636, -8.1882133, 8.1863842
41: -26.7388039, -6.9871507, -26.7540951, -6.9467654, -11.2957497, 11.2532196
42: -14.5370569, -2.1029325, -14.5442867, -2.1130047, -8.4998398, 8.5295353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0857812, upper bound: 5.0815405
time: 18.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0908519, upper bound: 5.0815405
time: 17.04 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.2068634, 0.6903830, -20.2380829, 0.7646816, -13.4762650, 13.4237518
1: -6.4198465, 5.2775564, -6.4441919, 5.3264289, -6.4334221, 6.4093075
2: -11.0078735, 2.2163849, -11.0144453, 2.2787886, -8.5967064, 8.5347481
3: -12.3135662, 3.4057605, -12.3322659, 3.4523709, -11.1770096, 11.1366348
4: -22.0202980, -5.6534400, -22.0383511, -5.5802140, -9.2765656, 9.2287560
5: -10.7809010, 5.6409879, -10.7927132, 5.7194977, -12.2171326, 12.1351242
6: -22.4729862, -4.5014038, -22.4892273, -4.4314308, -11.2124023, 11.1449966
7: -9.5426893, 8.9144630, -9.5601912, 8.9722338, -12.4923477, 12.4605637
8: -26.3269081, -5.6175342, -26.3539257, -5.5498505, -9.8533249, 9.8027115
9: -14.5681953, 2.1696248, -14.5971775, 2.1994603, -12.9336090, 12.9685173
10: -5.8815556, 11.7337246, -5.9166069, 11.7576218, -13.2333107, 13.2603111
11: 9.5923214, 21.0934906, 9.5229034, 21.1043739, -7.4627171, 7.5177174
12: -15.1357794, 9.8293285, -15.1905661, 9.8671551, -18.4855347, 18.4424820
13: -28.0193214, -3.0224547, -28.0390701, -3.0135849, -12.9099426, 12.9315491
14: -31.3686142, 0.6448498, -31.4066391, 0.6729064, -21.5007935, 21.5232315
15: -24.9233494, -10.6922369, -24.9457798, -10.6836271, -8.8692551, 8.8936195
16: -6.9356089, 7.8961754, -6.9522076, 7.9379535, -10.1479568, 10.1291180
17: -14.7595615, 11.7274294, -14.7689266, 11.7678967, -21.7262726, 21.6862946
18: -0.9091780, 12.5588903, -0.9280152, 12.5754051, -10.7100754, 10.6918716
19: -5.2955313, 4.7342639, -5.3218117, 4.7403741, -7.5710468, 7.5908413
20: -3.3947387, 7.9619222, -3.4350948, 7.9878263, -10.2516556, 10.2888260
21: -1.9400122, 8.8737640, -1.9787688, 8.8833017, -8.9082413, 8.9519768
22: -9.2178183, 2.7771027, -9.2456570, 2.7803192, -8.6678200, 8.7008152
23: 1.3658237, 12.4953451, 1.2967253, 12.5020275, -7.7054234, 7.7711983
24: -2.6766264, 10.4838657, -2.7485452, 10.4900970, -8.1369438, 8.2141304
25: 0.3925567, 13.7429943, 0.2786131, 13.7525320, -9.3291988, 9.4533958
26: -17.4042130, 2.4682031, -17.4510117, 2.4805458, -14.5641594, 14.6151962
27: -10.3108921, 6.2829418, -10.3114700, 6.2939034, -9.1542206, 9.1436577
28: 1.0891211, 13.5689487, 1.0345845, 13.5750008, -9.6168671, 9.6673965
29: -5.1050806, 8.3438988, -5.1302056, 8.3530025, -8.6082001, 8.6305695
30: 6.0014458, 17.7028446, 5.9205084, 17.7095127, -7.6299095, 7.7141495
31: -3.3777061, 10.3803082, -3.4166267, 10.3895626, -9.1469460, 9.1807137
32: -19.5819664, -2.7530036, -19.5961361, -2.6985989, -10.7501106, 10.6969318
33: -47.0194244, -21.5410080, -47.0403976, -21.5384216, -14.5121117, 14.5783157
34: -29.7056408, -10.5862484, -29.7283192, -10.5788326, -10.6758766, 10.6587067
35: -29.2184830, -9.9529095, -29.2363853, -9.9437618, -10.7119102, 10.7001991
36: -31.8819370, -9.4050331, -31.9024315, -9.3816013, -12.7360458, 12.7025261
37: -46.1226311, -23.4973373, -46.1391373, -23.4982357, -16.0348587, 16.0593567
38: -34.2931442, -11.5239496, -34.3177757, -11.4937325, -15.1469040, 15.1045647
39: -56.3038788, -30.6497879, -56.3207436, -30.6411629, -13.1711693, 13.2044258
40: -40.2390823, -23.2989960, -40.2491112, -23.2787476, -8.1827278, 8.1888027
41: -26.7351379, -6.9979944, -26.7531662, -6.9469786, -11.2925129, 11.2418556
42: -14.5356359, -2.1035388, -14.5453434, -2.1131210, -8.4970570, 8.5303631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0857781, upper bound: 5.0794745
time: 21.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0908483, upper bound: 5.0794745
time: 19.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2158699, 0.7000513, -20.2407379, 0.7655282, -13.4940720, 13.4228897
1: -6.4243307, 5.2889633, -6.4459333, 5.3267455, -6.4383507, 6.4169731
2: -11.0166941, 2.2319741, -11.0186253, 2.2791927, -8.6027946, 8.5546951
3: -12.3261662, 3.4220271, -12.3378839, 3.4527378, -11.1873779, 11.1597977
4: -22.0336323, -5.6290789, -22.0458565, -5.5795841, -9.2850723, 9.2613525
5: -10.7923069, 5.6585560, -10.7983932, 5.7199202, -12.2252045, 12.1595688
6: -22.4867725, -4.4866724, -22.4954681, -4.4312658, -11.2246666, 11.1660728
7: -9.5537672, 8.9271393, -9.5651293, 8.9726191, -12.5000916, 12.4755325
8: -26.3422165, -5.6004272, -26.3611832, -5.5496621, -9.8628311, 9.8186607
9: -14.5764008, 2.1823816, -14.5996294, 2.2003458, -12.9424973, 12.9823990
10: -5.8946056, 11.7557878, -5.9210114, 11.7588243, -13.2455025, 13.2846680
11: 9.5712156, 21.1054993, 9.5211229, 21.1107483, -7.4924793, 7.5291538
12: -15.1439514, 9.8343544, -15.1915236, 9.8682880, -18.4920425, 18.4497375
13: -28.0319824, -3.0116522, -28.0395775, -3.0096543, -12.9265671, 12.9408913
14: -31.3926182, 0.6548848, -31.4082890, 0.6775475, -21.5355988, 21.5341339
15: -24.9261093, -10.6852331, -24.9457169, -10.6820774, -8.8741989, 8.8994656
16: -6.9452434, 7.9086342, -6.9559727, 7.9382601, -10.1564751, 10.1388035
17: -14.7790575, 11.7354374, -14.7698689, 11.7719002, -21.7512970, 21.6959076
18: -0.9177911, 12.5650654, -0.9287460, 12.5767355, -10.7239342, 10.6992683
19: -5.3073635, 4.7373924, -5.3225470, 4.7415738, -7.5864506, 7.5948067
20: -3.4042902, 7.9666510, -3.4363837, 7.9881959, -10.2612572, 10.2960892
21: -1.9487885, 8.8800726, -1.9799485, 8.8836040, -8.9195709, 8.9588757
22: -9.2294216, 2.7840812, -9.2460966, 2.7834864, -8.6820412, 8.7072124
23: 1.3425953, 12.5058670, 1.2960513, 12.5079031, -7.7351456, 7.7799911
24: -2.6953566, 10.4953938, -2.7488728, 10.4958973, -8.1634064, 8.2231750
25: 0.3627632, 13.7606239, 0.2779865, 13.7619286, -9.3689651, 9.4672203
26: -17.4202957, 2.4728079, -17.4518509, 2.4815233, -14.5827789, 14.6205826
27: -10.3200722, 6.2873182, -10.3122616, 6.2945175, -9.1646996, 9.1484261
28: 1.0660846, 13.5764437, 1.0336747, 13.5789509, -9.6448822, 9.6744576
29: -5.1208324, 8.3515043, -5.1307621, 8.3565769, -8.6277714, 8.6370888
30: 5.9864864, 17.7124443, 5.9195671, 17.7142544, -7.6489525, 7.7217503
31: -3.3934124, 10.3877583, -3.4174831, 10.3929653, -9.1667061, 9.1877861
32: -19.5936661, -2.7414606, -19.6005173, -2.6981540, -10.7616119, 10.7164497
33: -47.0256500, -21.5260277, -47.0419235, -21.5369186, -14.5187035, 14.5985756
34: -29.7134094, -10.5759068, -29.7310982, -10.5783005, -10.6826553, 10.6767044
35: -29.2229843, -9.9464827, -29.2372036, -9.9432049, -10.7165375, 10.7100639
36: -31.8941879, -9.3996296, -31.9032745, -9.3802872, -12.7454262, 12.7102356
37: -46.1323891, -23.4892159, -46.1401825, -23.4954453, -16.0493164, 16.0684814
38: -34.3013802, -11.5173492, -34.3191109, -11.4931202, -15.1553574, 15.1121368
39: -56.3079071, -30.6436310, -56.3214722, -30.6402798, -13.1741447, 13.2180443
40: -40.2477570, -23.2946663, -40.2530899, -23.2786369, -8.1906128, 8.1972847
41: -26.7451820, -6.9851294, -26.7570858, -6.9465318, -11.3024940, 11.2586212
42: -14.5434389, -2.1007905, -14.5466909, -2.1127625, -8.5046959, 8.5359344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 689

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0857812, upper bound: 5.0867754
time: 24.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0908519, upper bound: 5.0867754
time: 74.33 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 101.36 seconds
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0574838, upper bound: 5.0908049
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0624141, upper bound: 5.0908049
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0669220, upper bound: 5.0908049
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0719021, upper bound: 5.0908049
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0663335, upper bound: 5.0908049
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0712640, upper bound: 5.0908049
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0757550, upper bound: 5.0908049
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0807517, upper bound: 5.0908049
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0857781, upper bound: 5.0743069
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0908483, upper bound: 5.0743069
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0857812, upper bound: 5.0815405
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0908519, upper bound: 5.0815405
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0857781, upper bound: 5.0794745
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0908483, upper bound: 5.0794745
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0857812, upper bound: 5.0867754
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 101.36
Output dim: 11, lower bound: -5.0908519, upper bound: 5.0867754
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 101.36
Output dim: 11, lower bound: -5.0727807, upper bound: 5.0911285
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 101.36
Output dim: 11, lower bound: -5.0822661, upper bound: 5.0911285
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 101.36
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0786667
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 101.36
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0859254
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 101.36
Output dim: 11, lower bound: -5.0816475, upper bound: 5.0911285
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 101.36
Output dim: 11, lower bound: -5.0911246, upper bound: 5.0837803
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 101.36
Output dim: 11, lower bound: -5.0911284, upper bound: 5.0911285

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 29.40 + 1845.49 = 1874.89 seconds
