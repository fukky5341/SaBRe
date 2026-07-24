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
execution time: IAR + RelationalAnalysis = 2.75 + 26.45 = 29.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -5.0948664, upper bound: 5.0948664

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0946834, upper bound: 5.0854322
time: 19.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0854322, upper bound: 5.0946834
time: 18.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 37.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 37.88
Output dim: 11, lower bound: -5.0946834, upper bound: 5.0854322
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 37.88
Output dim: 11, lower bound: -5.0854322, upper bound: 5.0946834

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4549370, 13.4581718
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3910179, 6.3934288
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5558014, 8.5573158
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1457825, 11.1478844
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2384567, 9.2410126
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1569748, 12.1593170
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1371193, 11.1397552
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4745483, 12.4769402
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8179588, 9.8215446
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9296722, 12.9300613
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2184219, 13.2186737
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4748602, 7.4730759
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.5035019, 18.5028000
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8647385, 12.8624992
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4964752, 21.4965134
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8873672, 8.8872623
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1083832, 10.1101151
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7168427, 21.7166672
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6829872, 10.6829834
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5595398, 7.5594330
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2510948, 10.2511787
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9085045, 8.9090195
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6700268, 8.6689796
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7014027, 7.7003193
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1272163, 8.1258163
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3468323, 9.3449402
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5591354, 14.5576248
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1071224, 9.1070499
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6052017, 9.6047134
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5964565, 8.5952454
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6474590, 7.6463165
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1424179, 9.1419792
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6857681, 10.6863518
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5405922, 14.5409660
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6523743, 10.6538315
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7031136, 10.7032013
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6904602, 12.6899605
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0769730, 16.0774536
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1157990, 15.1169701
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1966705, 13.1966972
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1896973, 8.1906891
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2330055, 11.2345200
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5040836, 8.5046539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0925934, upper bound: 5.0823782
time: 9.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0915462, upper bound: 5.0833889
time: 19.19 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4581718, 13.4549370
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3934288, 6.3910179
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5573158, 8.5558014
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1478882, 11.1457825
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2410164, 9.2384567
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1593170, 12.1569748
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1397552, 11.1371231
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4769440, 12.4745522
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8215446, 9.8179550
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9300613, 12.9296684
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2186737, 13.2184181
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4730749, 7.4748592
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.5028000, 18.5035019
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8624954, 12.8647385
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4965057, 21.4964752
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8872604, 8.8873692
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1101151, 10.1083832
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7166595, 21.7168579
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6829834, 10.6829872
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5594330, 7.5595379
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2511787, 10.2510948
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9090195, 8.9085045
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6689777, 8.6700249
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7003193, 7.7014027
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1258125, 8.1272144
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3449364, 9.3468323
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5576248, 14.5591354
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1070461, 9.1071186
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6047134, 9.6051979
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5952473, 8.5964584
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6463146, 7.6474571
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1419792, 9.1424179
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6863518, 10.6857681
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5409660, 14.5405922
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6538277, 10.6523743
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7032013, 10.7031136
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6899643, 12.6904640
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0774536, 16.0769730
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1169662, 15.1158028
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1966934, 13.1966705
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1906891, 8.1896954
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2345200, 11.2330055
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5046558, 8.5040836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0833889, upper bound: 5.0915462
time: 18.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0823782, upper bound: 5.0925934
time: 21.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 42.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 42.53
Output dim: 11, lower bound: -5.0925934, upper bound: 5.0823782
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 42.53
Output dim: 11, lower bound: -5.0915462, upper bound: 5.0833889
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 42.53
Output dim: 11, lower bound: -5.0833889, upper bound: 5.0915462
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 42.53
Output dim: 11, lower bound: -5.0823782, upper bound: 5.0925934

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4279251, 13.4336357
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3609276, 6.3665905
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5303650, 8.5346870
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1088333, 11.1153908
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1986427, 9.2056236
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1314545, 12.1367950
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1267204, 11.1310272
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4261971, 12.4337692
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7548027, 9.7656250
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9399643, 12.9355507
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2053909, 13.2031174
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4723873, 7.4707298
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4315567, 18.4212875
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8362427, 12.8271904
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4267578, 21.4169312
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8749352, 8.8736649
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0845413, 10.0881939
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6985855, 21.6961212
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6835136, 10.6844673
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5590534, 7.5590019
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2555122, 10.2548904
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9057083, 8.9044704
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6450253, 8.6401787
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6993809, 7.6985188
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1121826, 8.1140766
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3418503, 9.3403778
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5122223, 14.5053864
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0866737, 9.0903397
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6050377, 9.6045647
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5555840, 8.5487823
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6461830, 7.6445732
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1131668, 9.1160889
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6855469, 10.6861305
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5272636, 14.5310974
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6338921, 10.6419411
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6997414, 10.6998405
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6746559, 12.6735725
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0786285, 16.0794907
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1328735, 15.1381569
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1829338, 13.1850014
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1535301, 8.1612625
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2193069, 11.2245445
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5202942, 8.5237503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0820991, upper bound: 5.0770779
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0872090, upper bound: 5.0820991
time: 26.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4303970, 13.4311600
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3641815, 6.3633366
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5331726, 8.5318832
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1132889, 11.1109352
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2030640, 9.2012024
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1344604, 12.1337891
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1283951, 11.1293564
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4313850, 12.4285851
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7620354, 9.7583923
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9351578, 12.9403572
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2028580, 13.2056465
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4725132, 7.4706059
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4219818, 18.4308548
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8294296, 12.8340034
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4168930, 21.4267960
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8737717, 8.8748302
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0864639, 10.0862713
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6962967, 21.6983871
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6844711, 10.6835098
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5591068, 7.5589504
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2548027, 10.2555962
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9039574, 8.9062233
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6412258, 8.6439781
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6996021, 7.6982994
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1154747, 8.1107845
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3422737, 9.3399582
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5068970, 14.5107117
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0904083, 9.0866032
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6050491, 9.6045494
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5499916, 8.5543747
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6457138, 7.6450424
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1165276, 9.1127281
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6855469, 10.6861267
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5307198, 14.5276413
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6404877, 10.6353455
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6997490, 10.6998329
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6740761, 12.6741524
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0790100, 16.0791092
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1369858, 15.1340370
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1849785, 13.1829605
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1602669, 8.1545258
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2230301, 11.2208176
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5231781, 8.5208645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0912639, upper bound: 5.0781242
time: 17.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0861356, upper bound: 5.0831098
time: 17.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4311600, 13.4303970
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3633385, 6.3641815
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5318832, 8.5331726
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1109314, 11.1132889
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2012024, 9.2030640
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1337891, 12.1344604
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1293564, 11.1283951
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4285851, 12.4313812
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7583923, 9.7620354
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9403610, 12.9351578
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2056503, 13.2028618
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4706059, 7.4725132
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4308548, 18.4219818
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8339996, 12.8294296
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4267960, 21.4168930
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8748322, 8.8737717
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0862732, 10.0864658
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6983719, 21.6963120
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6835098, 10.6844711
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5589504, 7.5591068
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2555962, 10.2548065
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9062233, 8.9039574
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6439762, 8.6412239
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6983013, 7.6996021
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1107864, 8.1154747
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3399582, 9.3422737
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5107117, 14.5068970
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0866013, 9.0904083
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6045494, 9.6050529
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5543747, 8.5499916
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6450424, 7.6457138
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1127281, 9.1165295
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6861267, 10.6855469
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5276413, 14.5307198
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6353455, 10.6404877
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6998329, 10.6997490
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6741524, 12.6740761
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0791016, 16.0790100
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1340408, 15.1369896
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1829567, 13.1849785
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1545258, 8.1602688
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2208176, 11.2230301
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5208664, 8.5231800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0831098, upper bound: 5.0861356
time: 21.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0781242, upper bound: 5.0912639
time: 20.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4336395, 13.4279251
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3665886, 6.3609276
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5346870, 8.5303650
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1153946, 11.1088333
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2056236, 9.1986427
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1367950, 12.1314545
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1310272, 11.1267242
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4337654, 12.4261932
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7656250, 9.7548027
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9355545, 12.9399681
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2031174, 13.2053909
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4707279, 7.4723892
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4212875, 18.4315567
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8271942, 12.8362389
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4169312, 21.4267578
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8736649, 8.8749352
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0881958, 10.0845432
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6961136, 21.6985703
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6844673, 10.6835136
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5590000, 7.5590534
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2548904, 10.2555122
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9044724, 8.9057083
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6401768, 8.6450233
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6985188, 7.6993828
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1140747, 8.1121826
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3403778, 9.3418503
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5053864, 14.5122223
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0903397, 9.0866718
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6045609, 9.6050377
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5487823, 8.5555840
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6445732, 7.6461849
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1160889, 9.1131668
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6861305, 10.6855450
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5310936, 14.5272675
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6419411, 10.6338921
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6998405, 10.6997414
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6735725, 12.6746559
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0794907, 16.0786209
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1381531, 15.1328697
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1850014, 13.1829376
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1612625, 8.1535320
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2245445, 11.2193069
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5237503, 8.5202942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0820991, upper bound: 5.0872090
time: 17.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0770779, upper bound: 5.0923122
time: 29.42 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 49.26 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 49.26
Output dim: 11, lower bound: -5.0820991, upper bound: 5.0770779
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 49.26
Output dim: 11, lower bound: -5.0872090, upper bound: 5.0820991
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 49.26
Output dim: 11, lower bound: -5.0912639, upper bound: 5.0781242
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 49.26
Output dim: 11, lower bound: -5.0861356, upper bound: 5.0831098
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 49.26
Output dim: 11, lower bound: -5.0831098, upper bound: 5.0861356
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 49.26
Output dim: 11, lower bound: -5.0781242, upper bound: 5.0912639
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 49.26
Output dim: 11, lower bound: -5.0820991, upper bound: 5.0872090
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 49.26
Output dim: 11, lower bound: -5.0770779, upper bound: 5.0923122

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4272614, 13.4305840
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3591805, 6.3594780
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5247116, 8.5262070
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1078262, 11.1066208
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1886444, 9.1909981
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1261139, 12.1279907
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1283760, 11.1292877
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4219017, 12.4216347
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7412148, 9.7442627
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9358368, 12.9399300
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2028313, 13.2052612
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4709377, 7.4681587
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4094315, 18.4130554
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8162231, 12.8151207
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4168854, 21.4267960
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8628464, 8.8671875
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0849800, 10.0839710
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6896820, 21.6888809
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6838722, 10.6847763
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5568905, 7.5577030
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2502785, 10.2526627
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9006310, 8.9039040
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6408138, 8.6437016
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6990032, 7.6975555
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1129036, 8.1092319
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3399620, 9.3366394
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5005875, 14.5064697
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0745506, 9.0755386
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6039505, 9.6029434
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5486183, 8.5527687
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6429901, 7.6401196
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1126633, 9.1103058
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6861992, 10.6860142
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5314331, 14.5274963
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6403618, 10.6351318
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6954193, 10.6939430
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6729622, 12.6729507
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0787506, 16.0794449
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1354370, 15.1352386
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1849022, 13.1829453
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1582794, 8.1544571
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2191925, 11.2183037
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5213470, 8.5184612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0910985, upper bound: 5.0769354
time: 18.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0897968, upper bound: 5.0778345
time: 31.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4305878, 13.4272614
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3594780, 6.3591785
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5262070, 8.5247116
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1066208, 11.1078262
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1910019, 9.1886444
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1279907, 12.1261139
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1292877, 11.1283798
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4216347, 12.4219017
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7442627, 9.7412148
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9399261, 12.9358330
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2052650, 13.2028313
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4681568, 7.4709396
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4130554, 18.4094315
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8151245, 12.8162270
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4267960, 21.4168854
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8671875, 8.8628445
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0839729, 10.0849781
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6888885, 21.6896744
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6847763, 10.6838760
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5577030, 7.5568905
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2526627, 10.2502785
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9039040, 8.9006329
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6437016, 8.6408157
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6975536, 7.6990032
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1092300, 8.1129017
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3366394, 9.3399620
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5064697, 14.5005798
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0755386, 9.0745525
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6029434, 9.6039505
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5527687, 8.5486183
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6401176, 7.6429901
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1103058, 9.1126633
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6860161, 10.6861992
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5274963, 14.5314331
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6351318, 10.6403618
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6939430, 10.6954193
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6729546, 12.6729622
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0794449, 16.0787506
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1352386, 15.1354370
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1829414, 13.1849060
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1544571, 8.1582794
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2183037, 11.2191925
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5184631, 8.5213470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0778345, upper bound: 5.0897968
time: 15.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0769355, upper bound: 5.0910985
time: 16.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4330597, 13.4247856
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3627319, 6.3559246
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5290108, 8.5219040
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1110764, 11.1033669
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1954231, 9.1842232
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1309967, 12.1231155
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1309586, 11.1267052
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4268227, 12.4167137
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7514954, 9.7339821
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9351196, 12.9406395
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2027321, 13.2053604
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4682827, 7.4708138
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4034882, 18.4189987
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8083115, 12.8230400
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4169312, 21.4267502
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8660202, 8.8640099
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0858955, 10.0830555
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6865997, 21.6919403
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6857338, 10.6829147
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5577564, 7.5568390
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2519569, 10.2509842
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9021530, 8.9023838
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6399021, 8.6446133
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6977749, 7.6987839
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1125221, 8.1096115
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3370590, 9.3395424
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5011444, 14.5059052
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0792732, 9.0708160
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6029587, 9.6039391
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5471764, 8.5542107
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6396484, 7.6434593
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1136665, 9.1093025
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6860161, 10.6861954
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5309486, 14.5279808
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6417274, 10.6337662
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6939507, 10.6954117
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6723747, 12.6735420
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0798264, 16.0783691
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1393585, 15.1313210
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1849861, 13.1828613
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1611938, 8.1515427
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2220268, 11.2154694
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5213470, 8.5184631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0767876, upper bound: 5.0908615
time: 19.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0758767, upper bound: 5.0921463
time: 20.28 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 42.11 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.11
Output dim: 11, lower bound: -5.0910985, upper bound: 5.0769354
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.11
Output dim: 11, lower bound: -5.0897968, upper bound: 5.0778345
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.11
Output dim: 11, lower bound: -5.0778345, upper bound: 5.0897968
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.11
Output dim: 11, lower bound: -5.0769355, upper bound: 5.0910985
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.11
Output dim: 11, lower bound: -5.0767876, upper bound: 5.0908615
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.11
Output dim: 11, lower bound: -5.0758767, upper bound: 5.0921463

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4313965, 13.4353790
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3567619, 6.3573532
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5161133, 8.5189590
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1049843, 11.1038284
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1693649, 9.1740837
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1181870, 12.1211319
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1234818, 11.1245041
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4104309, 12.4111710
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7180481, 9.7239380
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9431076, 12.9458351
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2093887, 13.2105827
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4696665, 7.4665241
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3883820, 18.3890686
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7932663, 12.7891006
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4052582, 21.4135818
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8516273, 8.8574181
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0798111, 10.0776863
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6706848, 21.6672516
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6909752, 10.6935196
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5544300, 7.5558987
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2390900, 10.2428131
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8966789, 8.9003029
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6393356, 8.6423836
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6992435, 7.6979408
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1059895, 8.1031837
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3348179, 9.3307152
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4909554, 14.4984283
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0477505, 9.0519943
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6030312, 9.6018486
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5474110, 8.5515518
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6449966, 7.6410637
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1105690, 9.1088924
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6856575, 10.6854591
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5331917, 14.5290794
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6405678, 10.6353226
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6892090, 10.6872635
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6767349, 12.6776733
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0834351, 16.0852051
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1479721, 15.1505470
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1827393, 13.1810265
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1597404, 8.1573563
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2067108, 11.2074661
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5190258, 8.5160809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0900202, upper bound: 5.0683890
time: 19.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0825519, upper bound: 5.0758557
time: 18.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4319839, 13.4347229
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3570137, 6.3570614
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5174637, 8.5176086
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1050148, 11.1037750
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1717262, 9.1717224
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1191940, 12.1200714
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1235847, 11.1243896
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4113464, 12.4101639
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7208900, 9.7210960
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9417343, 12.9472008
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2081528, 13.2118187
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4693041, 7.4668694
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3854446, 18.3920059
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7902069, 12.7921562
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4036713, 21.4151688
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8530769, 8.8559685
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0786896, 10.0788078
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6680603, 21.6698990
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6926193, 10.6918793
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5550861, 7.5552406
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2404289, 10.2414780
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8970337, 8.8999519
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6394958, 8.6422234
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6993847, 7.6977959
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1068554, 8.1023178
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3340397, 9.3314972
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4925423, 14.4968414
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0510082, 9.0487347
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6028557, 9.6020241
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5473995, 8.5515633
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6439362, 7.6421223
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1112518, 9.1082115
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6856422, 10.6854725
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5330162, 14.5292206
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6405487, 10.6353378
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6887398, 10.6876717
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6776810, 12.6767273
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0845184, 16.0841293
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1507492, 15.1477699
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1829910, 13.1807785
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1611824, 8.1559162
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2083549, 11.2058258
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5189648, 8.5161381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0887125, upper bound: 5.0692829
time: 19.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0812645, upper bound: 5.0767563
time: 12.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4347229, 13.4319839
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3570595, 6.3570137
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5176086, 8.5174637
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1037788, 11.1050110
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1717224, 9.1717262
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1200638, 12.1191940
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1243896, 11.1235886
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4101639, 12.4113426
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7210960, 9.7208939
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9471970, 12.9417381
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2118149, 13.2081528
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4668703, 7.4693031
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3920059, 18.3854446
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7921524, 12.7902069
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4151611, 21.4036713
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8559685, 8.8530750
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0788116, 10.0786896
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6698914, 21.6680527
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6918793, 10.6926193
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5552387, 7.5550861
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2414780, 10.2404289
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8999519, 8.8970337
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6422234, 8.6394958
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6977940, 7.6993866
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1023159, 8.1068554
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3314953, 9.3340416
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4968376, 14.4925385
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0487347, 9.0510082
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6020241, 9.6028557
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5515614, 8.5474014
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6421204, 7.6439342
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1082115, 9.1112518
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6854706, 10.6856441
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5292168, 14.5330200
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6353378, 10.6405487
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6876717, 10.6887398
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6767273, 12.6776772
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0841293, 16.0845184
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1477661, 15.1507492
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1807785, 13.1829872
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1559143, 8.1611805
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2058258, 11.2083549
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5161381, 8.5189648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0767563, upper bound: 5.0812645
time: 27.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0692829, upper bound: 5.0887125
time: 16.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4353790, 13.4313965
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3573532, 6.3567619
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5189590, 8.5161133
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1038246, 11.1049805
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1740837, 9.1693649
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1211319, 12.1181870
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1245079, 11.1234818
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4111710, 12.4104309
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7239380, 9.7180481
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9458389, 12.9431038
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2105789, 13.2093887
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4665232, 7.4696655
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3890686, 18.3883820
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7891006, 12.7932625
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4135818, 21.4052582
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8574181, 8.8516273
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0776825, 10.0798111
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6672668, 21.6707001
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6935196, 10.6909752
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5558987, 7.5544300
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2428131, 10.2390900
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9003029, 8.8966808
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6423836, 8.6393356
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6979389, 7.6992435
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1031857, 8.1059875
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3307133, 9.3348198
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4984245, 14.4909515
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0519924, 9.0477486
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6018486, 9.6030350
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5515499, 8.5474129
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6410675, 7.6449928
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1088943, 9.1105690
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6854591, 10.6856575
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5290794, 14.5331955
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6353226, 10.6405640
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6872597, 10.6892090
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6776733, 12.6767387
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0852051, 16.0834351
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1505432, 15.1479721
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1810226, 13.1827393
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1573563, 8.1597404
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2074661, 11.2067108
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5160809, 8.5190258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0758557, upper bound: 5.0825519
time: 20.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0900202
time: 21.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4371948, 13.4295120
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3603172, 6.3537598
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5204124, 8.5146561
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1082344, 11.1005516
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1761436, 9.1673050
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1230698, 12.1161880
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1260643, 11.1219139
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4153519, 12.4061584
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7283287, 9.7136612
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9423904, 12.9465446
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2092896, 13.2106819
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4669924, 7.4691792
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3824387, 18.3950119
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7853470, 12.7970200
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4052963, 21.4135361
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8548050, 8.8542404
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0807343, 10.0767670
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6676331, 21.6703110
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6928368, 10.6916618
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5552921, 7.5550346
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2407684, 10.2411346
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8981972, 8.8987846
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6384239, 8.6432953
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6980152, 7.6991673
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1056080, 8.1035633
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3319187, 9.3336182
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4915123, 14.4978638
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0524731, 9.0472717
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6020393, 9.6028442
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5459728, 8.5529938
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6416550, 7.6444054
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1115723, 9.1078911
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6854744, 10.6856403
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5326729, 14.5295639
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6419334, 10.6339531
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6876793, 10.6887321
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6761475, 12.6782570
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0845108, 16.0841370
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1518860, 15.1466293
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1828232, 13.1809425
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1626511, 8.1544437
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2095490, 11.2046318
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5190220, 8.5160809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0757104, upper bound: 5.0823191
time: 15.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0682415, upper bound: 5.0897771
time: 16.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4378510, 13.4289246
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3606071, 6.3535080
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5217590, 8.5133095
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1082802, 11.1005249
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1785049, 9.1649399
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1241379, 12.1151886
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1261787, 11.1218109
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4163589, 12.4052467
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7311707, 9.7108154
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9410324, 12.9479103
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2080536, 13.2119179
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4666491, 7.4695415
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3795013, 18.3979492
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7822952, 12.8000755
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4037170, 21.4151230
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8562508, 8.8527908
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0796051, 10.0778885
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6649780, 21.6729660
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6944771, 10.6900177
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5559483, 7.5543766
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2421074, 10.2397957
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8985519, 8.8984318
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6385841, 8.6431351
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6981564, 7.6990242
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1064777, 8.1026955
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3311367, 9.3343964
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4930992, 14.4962769
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0557308, 9.0440140
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6018639, 9.6030197
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5459614, 8.5530014
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6405945, 7.6454620
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1122551, 9.1072083
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6854630, 10.6856537
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5325317, 14.5297394
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6419182, 10.6339722
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6872673, 10.6892014
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6770935, 12.6773186
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0855865, 16.0830536
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1546631, 15.1438522
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1830673, 13.1806984
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1640930, 8.1530037
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2111893, 11.2029877
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5189648, 8.5161400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0747937, upper bound: 5.0835956
time: 21.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0910666
time: 20.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 44.16 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0900202, upper bound: 5.0683890
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0825519, upper bound: 5.0758557
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0887125, upper bound: 5.0692829
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0812645, upper bound: 5.0767563
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0767563, upper bound: 5.0812645
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0692829, upper bound: 5.0887125
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0758557, upper bound: 5.0825519
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0900202
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0757104, upper bound: 5.0823191
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0682415, upper bound: 5.0897771
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0747937, upper bound: 5.0835956
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.16
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0910666

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4300613, 13.4345169
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3558159, 6.3554440
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5164909, 8.5183258
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0952873, 11.0968056
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1690025, 9.1734924
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1179276, 12.1209412
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1223068, 11.1237335
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4116745, 12.4102898
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7170525, 9.7213249
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9228363, 12.9306221
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2029190, 13.2056770
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4588604, 7.4517059
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3870850, 18.3915939
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7619705, 12.7682114
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4024811, 21.4101410
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8493118, 8.8561268
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0755157, 10.0745678
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6705322, 21.6669464
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6746063, 10.6706390
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5486984, 7.5468941
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2410774, 10.2420502
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8971748, 8.8973045
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6393299, 8.6423759
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6933613, 7.6894474
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0955887, 8.0877895
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3324337, 9.3276749
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4833336, 14.4876099
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0391254, 9.0355129
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6001511, 9.5968475
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5430031, 8.5456963
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6417656, 7.6353245
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1070137, 9.1010933
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6729584, 10.6765518
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5101128, 14.5135574
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6330872, 10.6334763
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6821671, 10.6864128
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6723022, 12.6773567
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0807495, 16.0858002
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1462097, 15.1499863
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1562881, 13.1622353
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1502876, 8.1503868
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2052803, 11.2071915
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5116253, 8.5118523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0899262, upper bound: 5.0640107
time: 15.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0842643, upper bound: 5.0680835
time: 20.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4345169, 13.4300652
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3554420, 6.3558159
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5183258, 8.5164909
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0968056, 11.0952873
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1734924, 9.1690025
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1209412, 12.1179276
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1237335, 11.1223068
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4102859, 12.4116707
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7213249, 9.7170525
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9306259, 12.9228363
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2056808, 13.2029228
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4517078, 7.4588585
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3915939, 18.3870850
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7682114, 12.7619667
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4101410, 21.4024811
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8561287, 8.8493118
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0745697, 10.0755138
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6669464, 21.6705322
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6706390, 10.6746063
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5468941, 7.5486984
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2420502, 10.2410774
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8973045, 8.8971748
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6423740, 8.6393299
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6894474, 7.6933594
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0877914, 8.0955906
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3276730, 9.3324280
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4876060, 14.4833374
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0355129, 9.0391273
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5968475, 9.6001511
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5456963, 8.5429993
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6353226, 7.6417656
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1010933, 9.1070137
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6765518, 10.6729584
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5135574, 14.5101128
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6334763, 10.6330872
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6864128, 10.6821671
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6773529, 12.6723061
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0858002, 16.0807495
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1499863, 15.1462059
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1622391, 13.1562843
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1503868, 8.1502857
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2071915, 11.2052803
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5118542, 8.5116234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0680835, upper bound: 5.0842643
time: 26.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0640107, upper bound: 5.0899262
time: 22.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4363327, 13.4281769
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3584061, 6.3528137
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5197792, 8.5150337
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1012154, 11.0908585
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1755524, 9.1669426
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1228790, 12.1159286
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1252899, 11.1207390
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4144669, 12.4073944
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7257156, 9.7126617
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9271774, 12.9262772
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2043839, 13.2042160
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4521770, 7.4583721
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3849640, 18.3937225
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7644577, 12.7657242
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4018555, 21.4107590
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8535118, 8.8519249
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0776138, 10.0724659
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6673279, 21.6701508
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6699524, 10.6752930
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5462914, 7.5493011
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2400055, 10.2431221
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8951988, 8.8992805
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6384144, 8.6432896
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6895199, 7.6932850
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0902176, 8.0931664
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3288784, 9.3312302
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4806938, 14.4902496
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0359898, 9.0386486
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5970383, 9.5999603
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5401154, 8.5485802
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6359138, 7.6411781
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1037750, 9.1043339
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6765671, 10.6729393
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5171509, 14.5064850
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6400909, 10.6264763
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6868286, 10.6816940
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6758270, 12.6738243
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0850983, 16.0814514
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1513290, 15.1448669
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1640320, 13.1544876
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1556816, 8.1449909
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2092705, 11.2032013
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5147953, 8.5086784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0679379, upper bound: 5.0840216
time: 20.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0638842, upper bound: 5.0896818
time: 18.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4369888, 13.4275894
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3586998, 6.3525620
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5211296, 8.5136871
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1012611, 11.0908318
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1779175, 9.1645775
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1239471, 12.1149216
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1254044, 11.1206360
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4154739, 12.4064865
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7285576, 9.7098198
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9258194, 12.9276428
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2031479, 13.2054520
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4518299, 7.4587345
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3820267, 18.3966599
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7613983, 12.7687798
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4002762, 21.4123459
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8549614, 8.8504772
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0764923, 10.0735912
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6646881, 21.6727982
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6715965, 10.6736488
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5469475, 7.5486450
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2413445, 10.2417831
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8955498, 8.8989277
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6385746, 8.6431293
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6896648, 7.6931419
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0910797, 8.0922985
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3280926, 9.3320084
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4822807, 14.4886627
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0392513, 9.0353909
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5968628, 9.6001358
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5401039, 8.5485916
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6348534, 7.6422348
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1044540, 9.1036530
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6765556, 10.6729546
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5170097, 14.5066605
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6400719, 10.6264915
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6864204, 10.6821594
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6767731, 12.6728859
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0861816, 16.0803680
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1541061, 15.1420898
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1642761, 13.1542435
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1571236, 8.1435490
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2109146, 11.2015572
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5147381, 8.5087376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0670328, upper bound: 5.0853373
time: 19.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0629395, upper bound: 5.0909730
time: 8.82 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 30.46 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 30.46
Output dim: 11, lower bound: -5.0899262, upper bound: 5.0640107
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 30.46
Output dim: 11, lower bound: -5.0842643, upper bound: 5.0680835
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 30.46
Output dim: 11, lower bound: -5.0680835, upper bound: 5.0842643
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 30.46
Output dim: 11, lower bound: -5.0640107, upper bound: 5.0899262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 30.46
Output dim: 11, lower bound: -5.0679379, upper bound: 5.0840216
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 30.46
Output dim: 11, lower bound: -5.0638842, upper bound: 5.0896818
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 30.46
Output dim: 11, lower bound: -5.0670328, upper bound: 5.0853373
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 30.46
Output dim: 11, lower bound: -5.0629395, upper bound: 5.0909730

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4280205, 13.4335823
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3520546, 6.3521519
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5028839, 8.5064487
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0880814, 11.0902519
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1455765, 9.1529732
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1025314, 12.1074219
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1200562, 11.1211090
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4015198, 12.4013901
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.6816826, 9.6905060
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9342575, 12.9405861
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2033081, 13.2060432
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4577351, 7.4503784
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3597794, 18.3604279
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7368813, 12.7388458
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4020538, 21.4097137
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8237915, 8.8337669
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0795517, 10.0774364
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6508331, 21.6456680
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6889153, 10.6864395
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5464973, 7.5456352
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2370377, 10.2389450
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8876991, 8.8884659
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6371040, 8.6405067
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6992607, 7.6958656
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0967178, 8.0902081
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3297062, 9.3246002
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4810257, 14.4874191
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0191345, 9.0191994
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5968361, 9.5930748
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5435257, 8.5461693
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6369133, 7.6291008
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1070061, 9.1024780
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6774254, 10.6799049
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5117722, 14.5146523
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6348495, 10.6351891
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6781693, 10.6818428
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6738777, 12.6789093
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0931702, 16.0992737
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1621323, 15.1682091
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1542282, 13.1604843
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1591492, 8.1601334
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2029953, 11.2057343
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5116501, 8.5118675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1680

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0750032, upper bound: 5.0639418
time: 22.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0898684, upper bound: 5.0635728
time: 22.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4335823, 13.4280243
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3521538, 6.3520527
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5064507, 8.5028858
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0902557, 11.0880814
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1529732, 9.1455803
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1074219, 12.1025314
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1211090, 11.1200562
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4013901, 12.4015160
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.6905098, 9.6816826
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9405899, 12.9342537
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2060394, 13.2033081
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4503765, 7.4577351
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3604279, 18.3597794
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7388420, 12.7368774
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4097137, 21.4020538
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8337669, 8.8237896
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0774384, 10.0795498
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6456757, 21.6508408
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6864395, 10.6889153
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5456352, 7.5464954
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2389450, 10.2370377
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8884659, 8.8876972
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6405067, 8.6371040
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6958656, 7.6992626
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0902100, 8.0967159
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3245945, 9.3297081
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4874191, 14.4810257
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0192032, 9.0191345
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5930786, 9.5968361
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5461693, 8.5435257
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6291008, 7.6369133
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1024780, 9.1070061
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6799049, 10.6774254
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5146484, 14.5117760
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6351891, 10.6348457
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6818466, 10.6781654
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6789131, 12.6738739
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0992737, 16.0931702
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1682129, 15.1621284
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1604843, 13.1542282
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1601334, 8.1591492
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2057304, 11.2029953
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5118675, 8.5116501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1680

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0635728, upper bound: 5.0898684
time: 25.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0639418, upper bound: 5.0888095
time: 15.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4360542, 13.4255486
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3554039, 6.3487988
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5092545, 8.5000820
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0947113, 11.0836258
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1573944, 9.1411552
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1104279, 12.0995293
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1227837, 11.1183853
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4065781, 12.3963280
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.6977425, 9.6744499
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9357834, 12.9390602
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2035141, 13.2058372
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4505024, 7.4576092
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3508530, 18.3693466
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7320366, 12.7436905
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.3998413, 21.4119186
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8326035, 8.8249550
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0793610, 10.0776272
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6433868, 21.6530991
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6873970, 10.6879578
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5456886, 7.5464439
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2382393, 10.2377434
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8867149, 8.8894501
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6367073, 8.6409035
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6960831, 7.6990433
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0934982, 8.0934238
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3250217, 9.3292847
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4820938, 14.4863510
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0229340, 9.0153980
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5930901, 9.5968246
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5405807, 8.5491142
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6286316, 7.6373825
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1058388, 9.1036453
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6799088, 10.6774216
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5181046, 14.5083199
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6417847, 10.6282539
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6818504, 10.6781616
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6783333, 12.6744537
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0996552, 16.0927887
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1723251, 15.1580086
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1625290, 13.1521873
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1668701, 8.1524124
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2094574, 11.1992722
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5147514, 8.5087643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1680

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0625079, upper bound: 5.0909147
time: 17.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0628717, upper bound: 5.0898428
time: 19.39 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 38.78 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 38.78
Output dim: 11, lower bound: -5.0750032, upper bound: 5.0639418
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 38.78
Output dim: 11, lower bound: -5.0898684, upper bound: 5.0635728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 38.78
Output dim: 11, lower bound: -5.0635728, upper bound: 5.0898684
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 38.78
Output dim: 11, lower bound: -5.0639418, upper bound: 5.0888095
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 38.78
Output dim: 11, lower bound: -5.0625079, upper bound: 5.0909147
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 38.78
Output dim: 11, lower bound: -5.0628717, upper bound: 5.0898428

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4164581, 13.4212952
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3480244, 6.3479958
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5023079, 8.5057945
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0892792, 11.0915222
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1452293, 9.1526337
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1050720, 12.1100006
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1035805, 11.1066933
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4033165, 12.4032593
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.6694069, 9.6775475
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9269409, 12.9327202
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1972733, 13.1992645
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4561396, 7.4485435
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3488083, 18.3504868
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7375450, 12.7393799
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.3946304, 21.4013443
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8129044, 8.8213272
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0781212, 10.0759544
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6475220, 21.6419525
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6863937, 10.6843910
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5475311, 7.5471210
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2351799, 10.2372932
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8845673, 8.8860550
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6379070, 8.6412640
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6957054, 7.6918030
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0955086, 8.0888290
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3220215, 9.3158112
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4783707, 14.4843826
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0184708, 9.0185871
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5971985, 9.5934296
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5425758, 8.5450859
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6368999, 7.6290741
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1054077, 9.1011467
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6636086, 10.6678143
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4960861, 14.5009232
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6233292, 10.6251068
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6626625, 10.6682777
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6641464, 12.6700554
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0952454, 16.1021347
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1507111, 15.1580963
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1399384, 13.1479797
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1534119, 8.1551132
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1909447, 11.1951866
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5112114, 8.5110512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0888310, upper bound: 5.0584109
time: 21.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0847229, upper bound: 5.0625243
time: 18.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4212952, 13.4164581
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3479939, 6.3480263
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5057945, 8.5023117
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0915222, 11.0892792
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1526375, 9.1452293
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1100006, 12.1050758
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1066933, 11.1035805
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4032555, 12.4033203
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.6775475, 9.6694031
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9327240, 12.9269409
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1992645, 13.1972771
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4485445, 7.4561405
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3504868, 18.3488083
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7393837, 12.7375450
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4013443, 21.3946304
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8213272, 8.8129044
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0759544, 10.0781212
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6419373, 21.6475220
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6843910, 10.6863937
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5471230, 7.5475292
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2372932, 10.2351799
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8860550, 8.8845673
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6412640, 8.6379032
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6918030, 7.6957054
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0888290, 8.0955086
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3158112, 9.3220215
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4843826, 14.4783707
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0185890, 9.0184708
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5934296, 9.5971985
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5450859, 8.5425758
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6290760, 7.6369019
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1011467, 9.1054077
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6678162, 10.6636066
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5009232, 14.4960861
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6251106, 10.6233292
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6682777, 10.6626625
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6700516, 12.6641502
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.1021347, 16.0952454
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1580963, 15.1507111
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1479797, 13.1399384
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1551132, 8.1534119
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1951866, 11.1909447
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5110512, 8.5112133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0625243, upper bound: 5.0847229
time: 19.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0584109, upper bound: 5.0888310
time: 19.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4237671, 13.4139862
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3512516, 6.3447723
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5085983, 8.4995041
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0959778, 11.0848236
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1570587, 9.1408081
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1130066, 12.1020737
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1083679, 11.1019058
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4084435, 12.3981361
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.6847801, 9.6621704
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9279099, 12.9317474
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1967392, 13.1998062
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4486666, 7.4560165
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3409195, 18.3583755
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7325706, 12.7443581
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.3914795, 21.4044952
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8201599, 8.8140697
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0778770, 10.0761986
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6396790, 21.6497879
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6853485, 10.6854362
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5471725, 7.5474777
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2365837, 10.2358856
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8843040, 8.8863182
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6374645, 8.6417027
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6920204, 7.6954880
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0921211, 8.0922165
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3162308, 9.3215942
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4790573, 14.4836960
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0223236, 9.0147343
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5934448, 9.5971832
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5394936, 8.5481644
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6286068, 7.6373711
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1045074, 9.1020470
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6678200, 10.6636047
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5043793, 14.4926338
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6317024, 10.6167336
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6682854, 10.6626587
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6694717, 12.6647301
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.1025238, 16.0948639
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1622162, 15.1465950
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1500244, 13.1378937
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1618500, 8.1466751
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1989136, 11.1872215
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5139351, 8.5083275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0614576, upper bound: 5.0857870
time: 22.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0573344, upper bound: 5.0898805
time: 19.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4243546, 13.4132614
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3513050, 6.3446426
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5086632, 8.4994240
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0959091, 11.0848923
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1570473, 9.1408195
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1129379, 12.1021080
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1063080, 11.1039658
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4083595, 12.3981972
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.6852722, 9.6614914
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9284668, 12.9311905
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1974564, 13.1990585
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4489069, 7.4557743
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3398819, 18.3592834
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7327003, 12.7442169
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.3924179, 21.4035492
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8217087, 8.8125153
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0779305, 10.0761452
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6400757, 21.6493912
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6848755, 10.6859093
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5467224, 7.5479259
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2363815, 10.2360802
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8835831, 8.8870239
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6375103, 8.6416454
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6925278, 7.6949806
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0922928, 8.0920448
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3173294, 9.3204994
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4794388, 14.4833145
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0222740, 9.0147858
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5934525, 9.5971603
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5396309, 8.5480309
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6286182, 7.6373577
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1042404, 9.1022873
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6660919, 10.6653309
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5024185, 14.4945946
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6302643, 10.6181717
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6663475, 10.6645927
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6686020, 12.6654816
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.1017227, 16.0956573
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1609116, 15.1478958
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1482391, 13.1396561
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1611328, 8.1473923
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1974030, 11.1887245
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5143166, 8.5079498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1401

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0618177, upper bound: 5.0847370
time: 17.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0576869, upper bound: 5.0888199
time: 17.76 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 37.26 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 37.26
Output dim: 11, lower bound: -5.0888310, upper bound: 5.0584109
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 37.26
Output dim: 11, lower bound: -5.0847229, upper bound: 5.0625243
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 37.26
Output dim: 11, lower bound: -5.0625243, upper bound: 5.0847229
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 37.26
Output dim: 11, lower bound: -5.0584109, upper bound: 5.0888310
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 37.26
Output dim: 11, lower bound: -5.0614576, upper bound: 5.0857870
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 37.26
Output dim: 11, lower bound: -5.0573344, upper bound: 5.0898805
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 37.26
Output dim: 11, lower bound: -5.0618177, upper bound: 5.0847370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 37.26
Output dim: 11, lower bound: -5.0576869, upper bound: 5.0888199

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4152908, 13.4041595
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3483315, 6.3415070
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5065651, 8.4972534
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0882111, 11.0762405
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1522026, 9.1354408
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1130219, 12.1020851
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1083183, 11.1016769
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4077721, 12.3974075
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.6742096, 9.6499290
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9191895, 12.9226799
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1940918, 13.1973305
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4395428, 7.4485188
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3360901, 18.3539734
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7320671, 12.7439079
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.3804016, 21.3943787
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8102684, 8.8047943
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0727806, 10.0706196
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6279907, 21.6397171
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6746864, 10.6752548
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5463104, 7.5469418
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2369041, 10.2364159
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8839455, 8.8865147
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6359310, 8.6403427
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6896553, 7.6933899
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0869751, 8.0876884
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3133392, 9.3188858
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4723587, 14.4776306
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0211544, 9.0133705
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5923843, 9.5962677
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5319538, 8.5415611
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6227760, 7.6332779
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1044731, 9.1018295
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6616707, 10.6565132
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4865837, 14.4725761
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6149292, 10.5981140
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6636276, 10.6570168
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6701775, 12.6654854
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.1064682, 16.0969315
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1644592, 15.1481514
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1351891, 13.1213188
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1616249, 8.1442070
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1978416, 11.1848679
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5153980, 8.5081825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0560373, upper bound: 5.0895569
time: 18.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0570091, upper bound: 5.0885995
time: 9.11 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 29.62 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 29.62
Output dim: 11, lower bound: -5.0560373, upper bound: 5.0895569
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 29.62
Output dim: 11, lower bound: -5.0570091, upper bound: 5.0885995

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 29.20 + 1134.60 = 1163.80 seconds
