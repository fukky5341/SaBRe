## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 1800 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.76 + 27.55 = 30.30 seconds
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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0946834, upper bound: 5.0854322
time: 19.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0854322, upper bound: 5.0946834
time: 19.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 39.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 39.32
Output dim: 11, lower bound: -5.0946834, upper bound: 5.0854322
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 39.32
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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0925934, upper bound: 5.0823782
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0915462, upper bound: 5.0833889
time: 20.04 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0833889, upper bound: 5.0915462
time: 19.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0823782, upper bound: 5.0925934
time: 22.26 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 44.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 44.27
Output dim: 11, lower bound: -5.0925934, upper bound: 5.0823782
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 44.27
Output dim: 11, lower bound: -5.0915462, upper bound: 5.0833889
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 44.27
Output dim: 11, lower bound: -5.0833889, upper bound: 5.0915462
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 44.27
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

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0820991, upper bound: 5.0770779
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0872090, upper bound: 5.0820991
time: 27.83 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0912639, upper bound: 5.0781242
time: 18.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0861356, upper bound: 5.0831098
time: 17.88 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0831098, upper bound: 5.0861356
time: 21.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0781242, upper bound: 5.0912639
time: 20.64 seconds

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

Time for backsubstitution: 2.17 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0820991, upper bound: 5.0872090
time: 18.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0770779, upper bound: 5.0923122
time: 30.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 50.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 50.72
Output dim: 11, lower bound: -5.0820991, upper bound: 5.0770779
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 50.72
Output dim: 11, lower bound: -5.0872090, upper bound: 5.0820991
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 50.72
Output dim: 11, lower bound: -5.0912639, upper bound: 5.0781242
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 50.72
Output dim: 11, lower bound: -5.0861356, upper bound: 5.0831098
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 50.72
Output dim: 11, lower bound: -5.0831098, upper bound: 5.0861356
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 50.72
Output dim: 11, lower bound: -5.0781242, upper bound: 5.0912639
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 50.72
Output dim: 11, lower bound: -5.0820991, upper bound: 5.0872090
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 50.72
Output dim: 11, lower bound: -5.0770779, upper bound: 5.0923122

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4247894, 13.4330559
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3559227, 6.3627319
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5219040, 8.5290108
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1033707, 11.1110802
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1842232, 9.1954231
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1231079, 12.1309967
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1267052, 11.1309586
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4167137, 12.4268188
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7339821, 9.7514954
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9406433, 12.9351234
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2053566, 13.2027321
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4708118, 7.4682846
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4189987, 18.4034882
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8230438, 12.8083076
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4267502, 21.4169312
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8640060, 8.8660221
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0830498, 10.0858974
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6919403, 21.6866150
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6829147, 10.6857338
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5568371, 7.5577564
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2509842, 10.2519569
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9023857, 8.9021511
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6446133, 8.6399021
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6987858, 7.6977749
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1096115, 8.1125221
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3395424, 9.3370590
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5059128, 14.5011444
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0708160, 9.0792732
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6039391, 9.6029587
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5542068, 8.5471764
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6434593, 7.6396484
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1093025, 9.1136665
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6861954, 10.6860180
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5279808, 14.5309486
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6337662, 10.6417274
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6954155, 10.6939507
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6735420, 12.6723709
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0783691, 16.0798264
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1313171, 15.1393547
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1828651, 13.1849861
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1515427, 8.1611938
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2154694, 11.2220306
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5184631, 8.5213470

Time for backsubstitution: 2.14 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0921463, upper bound: 5.0758767
time: 20.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0908615, upper bound: 5.0767876
time: 11.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4273529, 13.4304962
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3570671, 6.3615875
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5246887, 8.5262299
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1045151, 11.1099243
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1884422, 9.1912041
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1256561, 12.1284561
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1266518, 11.1310120
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4192390, 12.4242935
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7406731, 9.7448044
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9395370, 12.9362259
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2050056, 13.2030869
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4699421, 7.4691544
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4137573, 18.4087296
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8173599, 12.8139877
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4267578, 21.4169235
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8672943, 8.8627377
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0822411, 10.0867023
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6890717, 21.6894836
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6847801, 10.6838722
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5578060, 7.5567875
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2525749, 10.2503624
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9033890, 8.9011459
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6447468, 8.6397686
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6986370, 7.6979198
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1106300, 8.1115036
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3385315, 9.3380661
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5079803, 14.4990692
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0756073, 9.0744820
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6034317, 9.6034660
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5539780, 8.5474091
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6412621, 7.6418495
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1107445, 9.1122227
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6854324, 10.6867828
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5271187, 14.5318108
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6336784, 10.6418152
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6938553, 10.6955109
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6734505, 12.6724625
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0789642, 16.0792313
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1340714, 15.1366043
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1829185, 13.1849289
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1534653, 8.1592731
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2167892, 11.2207069
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5178909, 8.5219173

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0869263, upper bound: 5.0806922
time: 17.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0859900, upper bound: 5.0819179
time: 19.21 seconds

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

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0910985, upper bound: 5.0769354
time: 18.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0897968, upper bound: 5.0778345
time: 32.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4298248, 13.4280205
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3603210, 6.3583336
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5274925, 8.5234222
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1089783, 11.1054688
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1928673, 9.1867790
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1286621, 12.1254501
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1283264, 11.1293411
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4244270, 12.4191055
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7479057, 9.7375717
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9347305, 12.9410324
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2024727, 13.2056160
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4700642, 7.4690323
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4041824, 18.4182968
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8105469, 12.8208008
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4168930, 21.4267883
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8661270, 8.8639030
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0841713, 10.0847797
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6868134, 21.6917496
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6857376, 10.6829147
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5578594, 7.5567341
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2518692, 10.2510719
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9016380, 8.9028988
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6409473, 8.6435680
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6988583, 7.6977024
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1139221, 8.1082134
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3389549, 9.3376465
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5026550, 14.5043945
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0793419, 9.0707474
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6034470, 9.6034508
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5483856, 8.5529976
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6407928, 7.6423187
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1141090, 9.1088619
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6854362, 10.6867790
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5305710, 14.5283585
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6402740, 10.6352196
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6938591, 10.6955032
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6728783, 12.6730423
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0793457, 16.0788498
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1381912, 15.1324883
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1849632, 13.1828880
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1602020, 8.1525364
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2205162, 11.2169838
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5207787, 8.5190315

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0858551, upper bound: 5.0817188
time: 15.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0849112, upper bound: 5.0829289
time: 18.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4280243, 13.4298210
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3583336, 6.3603210
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5234222, 8.5274925
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1054688, 11.1089783
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1867790, 9.1928673
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1254501, 12.1286621
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1293373, 11.1283264
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4191017, 12.4244308
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7375717, 9.7479057
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9410324, 12.9347305
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2056160, 13.2024803
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4690304, 7.4700661
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4182968, 18.4041824
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8208008, 12.8105507
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4267883, 21.4168930
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8639030, 8.8661270
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0847816, 10.0841694
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6917572, 21.6868057
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6829147, 10.6857376
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5567341, 7.5578594
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2510681, 10.2518692
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9028969, 8.9016380
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6435680, 8.6409492
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6977024, 7.6988564
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1082115, 8.1139221
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3376465, 9.3389549
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5044022, 14.5026550
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0707474, 9.0793438
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6034508, 9.6034431
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5529976, 8.5483856
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6423187, 7.6407909
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1088600, 9.1141071
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6867790, 10.6854343
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5283585, 14.5305710
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6352196, 10.6402740
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6955032, 10.6938629
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6730385, 12.6728745
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0788498, 16.0793457
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1324921, 15.1381874
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1828880, 13.1849632
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1525383, 8.1602001
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2169838, 11.2205124
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5190315, 8.5207767

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0829288, upper bound: 5.0849112
time: 25.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0817188, upper bound: 5.0858551
time: 22.28 seconds

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

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0778345, upper bound: 5.0897968
time: 16.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0769355, upper bound: 5.0910985
time: 16.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4304962, 13.4273491
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3615875, 6.3570671
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5262299, 8.5246887
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1099243, 11.1045189
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1912041, 9.1884422
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1284485, 12.1256561
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1310120, 11.1266556
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4242897, 12.4192429
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7448044, 9.7406731
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9362259, 12.9395370
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2030830, 13.2050056
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4691525, 7.4699440
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4087296, 18.4137573
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8139877, 12.8173599
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4169235, 21.4267578
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8627396, 8.8672924
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0867043, 10.0822468
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6894989, 21.6890717
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6838722, 10.6847801
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5567875, 7.5578079
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2503624, 10.2525787
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9011459, 8.9033890
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6397686, 8.6447487
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6979198, 7.6986389
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1115036, 8.1106300
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3380661, 9.3385315
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4990768, 14.5079803
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0744820, 9.0756073
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6034660, 9.6034317
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5474091, 8.5539780
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6418495, 7.6412601
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1122246, 9.1107464
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6867828, 10.6854305
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5318108, 14.5271187
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6418152, 10.6336784
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6955109, 10.6938553
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6724586, 12.6734543
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0792313, 16.0789642
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1366043, 15.1340714
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1849251, 13.1829185
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1592751, 8.1534634
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2207069, 11.2167892
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5219193, 8.5178909

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0819179, upper bound: 5.0859900
time: 21.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0806922, upper bound: 5.0869263
time: 18.71 seconds

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

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0767876, upper bound: 5.0908615
time: 20.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0758767, upper bound: 5.0921463
time: 20.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 43.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0921463, upper bound: 5.0758767
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0908615, upper bound: 5.0767876
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0869263, upper bound: 5.0806922
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0859900, upper bound: 5.0819179
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0910985, upper bound: 5.0769354
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0897968, upper bound: 5.0778345
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0858551, upper bound: 5.0817188
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0849112, upper bound: 5.0829289
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0829288, upper bound: 5.0849112
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0817188, upper bound: 5.0858551
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0778345, upper bound: 5.0897968
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0769355, upper bound: 5.0910985
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0819179, upper bound: 5.0859900
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0806922, upper bound: 5.0869263
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0767876, upper bound: 5.0908615
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 43.44
Output dim: 11, lower bound: -5.0758767, upper bound: 5.0921463

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4289246, 13.4378510
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3535080, 6.3606071
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5133095, 8.5217628
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1005287, 11.1082840
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1649399, 9.1785049
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1151810, 12.1241379
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1218071, 11.1261787
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4052429, 12.4163589
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7108154, 9.7311707
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9479141, 12.9410286
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2119217, 13.2080536
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4695406, 7.4666481
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3979492, 18.3795013
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8000717, 12.7822876
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4151230, 21.4037170
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8527908, 8.8562527
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0778885, 10.0796089
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6729736, 21.6649857
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6900177, 10.6944771
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5543766, 7.5559502
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2397957, 10.2421074
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8984337, 8.8985519
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6431351, 8.6385841
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6990261, 7.6981583
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1026974, 8.1064758
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3343983, 9.3311386
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4962807, 14.4931030
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0440121, 9.0557289
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6030197, 9.6018600
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5530033, 8.5459595
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6454620, 7.6405926
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1072083, 9.1122532
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6856537, 10.6854610
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5297394, 14.5325317
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6339722, 10.6419144
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6892014, 10.6872711
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6773148, 12.6770935
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0830536, 16.0855865
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1438522, 15.1546669
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1806946, 13.1830673
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1530037, 8.1640930
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2029877, 11.2111893
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5161381, 8.5189648

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0910666, upper bound: 5.0673420
time: 19.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0835956, upper bound: 5.0747936
time: 24.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4295120, 13.4371948
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3537598, 6.3603153
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5146561, 8.5204124
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1005516, 11.1082344
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1673050, 9.1761436
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1161880, 12.1230698
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1219139, 11.1260643
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4061584, 12.4153519
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7136574, 9.7283287
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9465408, 12.9423943
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2106857, 13.2092896
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4691782, 7.4669933
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3950119, 18.3824387
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7970200, 12.7853470
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4135361, 21.4052963
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8542404, 8.8548050
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0767670, 10.0807304
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6703186, 21.6676331
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6916618, 10.6928368
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5550327, 7.5552921
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2411346, 10.2407684
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8987846, 8.8981991
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6432953, 8.6384239
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6991673, 7.6980152
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1035633, 8.1056099
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3336201, 9.3319168
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4978676, 14.4915161
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0472698, 9.0524712
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6028404, 9.6020393
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5529919, 8.5459709
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6444016, 7.6416512
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1078911, 9.1115723
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6856422, 10.6854744
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5295639, 14.5326729
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6339531, 10.6419334
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6887321, 10.6876793
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6782608, 12.6761475
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0841370, 16.0845108
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1466293, 15.1518860
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1809464, 13.1828232
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1544456, 8.1626530
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2046318, 11.2095490
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5160809, 8.5190239

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0897771, upper bound: 5.0682415
time: 14.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0823190, upper bound: 5.0757104
time: 19.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4314804, 13.4352188
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3546486, 6.3594227
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5160904, 8.5189819
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1016731, 11.1071091
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1691628, 9.1742859
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1177292, 12.1215363
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1217575, 11.1262207
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4077759, 12.4137344
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7175064, 9.7244797
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9468079, 12.9421310
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2115631, 13.2084084
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4686518, 7.4675198
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3927078, 18.3847427
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7943954, 12.7879677
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4151306, 21.4037094
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8560753, 8.8529682
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0770798, 10.0804176
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6701050, 21.6678619
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6918793, 10.6926155
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5553455, 7.5549812
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2413902, 10.2405128
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8994370, 8.8975468
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6432686, 8.6384506
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6988773, 7.6983032
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1037159, 8.1054573
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3333912, 9.3321457
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4983482, 14.4910278
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0488033, 9.0509377
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6025124, 9.6023674
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5527706, 8.5461922
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6432648, 7.6427937
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1086502, 9.1108093
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6848869, 10.6862259
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5288429, 14.5333939
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6338844, 10.6420021
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6875839, 10.6888275
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6772308, 12.6771774
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0836487, 16.0849991
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1465988, 15.1519165
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1807556, 13.1830101
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1549225, 8.1621742
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2043114, 11.2098694
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5155697, 8.5195351

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0858396, upper bound: 5.0721293
time: 19.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0783962, upper bound: 5.0796168
time: 18.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4321442, 13.4346313
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3549423, 6.3591709
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5174408, 8.5176315
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1017265, 11.1070824
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1715240, 9.1719246
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1187973, 12.1205292
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1218719, 11.1261139
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4087830, 12.4128227
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7203484, 9.7216377
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9454422, 12.9434967
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2103271, 13.2096405
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4683084, 7.4678822
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3897705, 18.3876801
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7913437, 12.7910233
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4135437, 21.4052963
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8575249, 8.8515205
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0759583, 10.0815392
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6674500, 21.6705093
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6935234, 10.6909714
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5560017, 7.5543232
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2427292, 10.2391777
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8997917, 8.8971939
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6434288, 8.6382904
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6990223, 7.6981602
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1045818, 8.1045914
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3326092, 9.3329239
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4999352, 14.4894409
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0520611, 9.0476799
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6023369, 9.6025467
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5527592, 8.5462036
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6422043, 7.6438503
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1093330, 9.1101284
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6848755, 10.6862411
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5287018, 14.5335693
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6338692, 10.6420212
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6871719, 10.6892967
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6781769, 12.6762352
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0847244, 16.0839157
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1493835, 15.1491394
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1809998, 13.1827660
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1563644, 8.1607323
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2059517, 11.2082291
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5155125, 8.5195942

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0848997, upper bound: 5.0733404
time: 21.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0774672, upper bound: 5.0808508
time: 18.44 seconds

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

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0900202, upper bound: 5.0683890
time: 19.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0825519, upper bound: 5.0758557
time: 18.98 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0887125, upper bound: 5.0692829
time: 20.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0812645, upper bound: 5.0767563
time: 12.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4339600, 13.4327469
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3579063, 6.3561687
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5188980, 8.5161743
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1061363, 11.1026535
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1735840, 9.1698608
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1207352, 12.1185303
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1234283, 11.1245499
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4129639, 12.4085464
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7247391, 9.7172470
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9420013, 12.9469376
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2090378, 13.2109375
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4687777, 7.4673958
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3831329, 18.3943100
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7875900, 12.7947807
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4052582, 21.4135742
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8549080, 8.8541336
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0790024, 10.0784950
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6678162, 21.6701279
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6928368, 10.6916580
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5553989, 7.5549297
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2406845, 10.2412224
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8976860, 8.8992996
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6394691, 8.6422501
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6990948, 7.6980839
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1070080, 8.1021652
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3338108, 9.3317261
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4930229, 14.4963531
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0525417, 9.0472031
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6025276, 9.6023560
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5471783, 8.5517807
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6427917, 7.6432629
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1120148, 9.1074486
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6848946, 10.6862240
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5322952, 14.5299416
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6404800, 10.6354065
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6875877, 10.6888199
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6766510, 12.6777573
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0840302, 16.0846100
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1507187, 15.1477966
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1828003, 13.1809654
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1616592, 8.1554375
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2080345, 11.2061462
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5184536, 8.5166492

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0847673, upper bound: 5.0731411
time: 21.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0773290, upper bound: 5.0806484
time: 18.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4346161, 13.4321594
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3581963, 6.3559170
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5202446, 8.5148277
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1061821, 11.1026230
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1759491, 9.1674995
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1218033, 12.1175232
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1235466, 11.1244431
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4139709, 12.4076347
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7275810, 9.7144051
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9406357, 12.9483032
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2078018, 13.2121735
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4684305, 7.4677582
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3801956, 18.3972549
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7845230, 12.7978363
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4036789, 21.4151611
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8563576, 8.8526859
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0778809, 10.0796165
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6651917, 21.6727753
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6944809, 10.6900139
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5560551, 7.5542717
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2420197, 10.2398834
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8980370, 8.8989468
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6396294, 8.6420879
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6992397, 7.6979427
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1078739, 8.1012993
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3330326, 9.3325043
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4946098, 14.4947662
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0557995, 9.0439434
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6023521, 9.6025314
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5471706, 8.5517960
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6417389, 7.6443214
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1126938, 9.1067677
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6848793, 10.6862373
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5321541, 14.5301170
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6404610, 10.6354256
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6871796, 10.6892929
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6775970, 12.6768150
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0851135, 16.0835342
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1534958, 15.1450195
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1830444, 13.1807213
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1631012, 8.1539955
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2096786, 11.2045021
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5183964, 8.5167103

Time for backsubstitution: 2.19 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0838200, upper bound: 5.0743509
time: 24.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0763922, upper bound: 5.0818602
time: 17.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4321594, 13.4346161
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3559151, 6.3581963
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5148277, 8.5202446
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1026268, 11.1061821
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1674995, 9.1759491
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1175232, 12.1218033
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1244431, 11.1235428
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4076385, 12.4139709
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7144051, 9.7275848
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9483032, 12.9406357
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2121658, 13.2077980
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4677591, 7.4684315
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3972549, 18.3801956
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7978363, 12.7845306
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4151611, 21.4036789
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8526878, 8.8563576
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0796204, 10.0778809
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6727600, 21.6651764
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6900139, 10.6944809
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5542698, 7.5560551
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2398834, 10.2420197
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8989449, 8.8980370
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6420898, 8.6396294
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6979427, 7.6992416
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1012974, 8.1078739
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3325024, 9.3330307
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4947701, 14.4946136
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0439434, 9.0557995
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6025314, 9.6023483
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5517941, 8.5471687
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6443253, 7.6417351
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1067657, 9.1126938
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6862373, 10.6848793
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5301170, 14.5321579
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6354256, 10.6404610
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6892891, 10.6871796
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6768188, 12.6775970
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0835342, 16.0851135
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1450195, 15.1534996
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1807175, 13.1830444
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1539955, 8.1631012
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2045021, 11.2096786
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5167103, 8.5183964

Time for backsubstitution: 2.19 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0818602, upper bound: 5.0763922
time: 21.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0743509, upper bound: 5.0838200
time: 25.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4327469, 13.4339600
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3561668, 6.3579044
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5161743, 8.5188942
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1026497, 11.1061325
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1698608, 9.1735840
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1185303, 12.1207352
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1245461, 11.1234283
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4085464, 12.4129639
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7172470, 9.7247391
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9469376, 12.9420013
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2109375, 13.2090340
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4673967, 7.4687767
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3943100, 18.3831329
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7947769, 12.7875824
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4135742, 21.4052658
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8541336, 8.8549099
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0784912, 10.0790024
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6701355, 21.6678238
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6916580, 10.6928368
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5549297, 7.5553989
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2412224, 10.2406845
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8992996, 8.8976860
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6422501, 8.6394691
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6980839, 7.6990967
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1021671, 8.1070061
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3317242, 9.3338127
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4963570, 14.4930267
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0472012, 9.0525398
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6023521, 9.6025276
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5517826, 8.5471802
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6432648, 7.6427937
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1074486, 9.1120129
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6862221, 10.6848927
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5299416, 14.5322952
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6354103, 10.6404800
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6888199, 10.6875916
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6777573, 12.6766510
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0846100, 16.0840302
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1477966, 15.1507187
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1809692, 13.1827965
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1554375, 8.1616592
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2061462, 11.2080345
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5166492, 8.5184536

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0806484, upper bound: 5.0773290
time: 28.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0731411, upper bound: 5.0847673
time: 20.79 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0767563, upper bound: 5.0812645
time: 27.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0692829, upper bound: 5.0887125
time: 16.96 seconds

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

Time for backsubstitution: 2.19 seconds

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
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0758557, upper bound: 5.0825519
time: 20.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0900202
time: 22.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4346313, 13.4321442
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3591728, 6.3549423
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5176315, 8.5174408
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1070824, 11.1017265
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1719246, 9.1715240
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1205292, 12.1187973
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1261139, 11.1218719
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4128265, 12.4087830
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7216377, 9.7203522
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9434967, 12.9454422
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2096481, 13.2103271
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4678812, 7.4683075
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3876801, 18.3897705
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7910233, 12.7913437
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4052963, 21.4135437
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8515205, 8.8575230
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0815430, 10.0759583
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6705017, 21.6674423
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6909714, 10.6935234
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5543232, 7.5560017
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2391777, 10.2427292
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8971939, 8.8997898
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6382904, 8.6434288
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6981602, 7.6990223
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1045895, 8.1045818
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3329258, 9.3326111
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4894447, 14.4999390
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0476818, 9.0520630
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6025429, 9.6023369
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5462017, 8.5527611
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6438522, 7.6422062
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1101303, 9.1093330
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6862411, 10.6848755
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5335693, 14.5287018
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6420212, 10.6338654
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6892967, 10.6871719
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6762390, 12.6781769
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0839157, 16.0847244
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1491394, 15.1493797
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1827621, 13.1810036
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1607323, 8.1563644
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2082291, 11.2059555
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5195942, 8.5155106

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0808508, upper bound: 5.0774672
time: 18.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0733404, upper bound: 5.0848997
time: 27.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4352188, 13.4314842
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3594246, 6.3546505
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5189819, 8.5160904
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1071053, 11.1016769
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1742859, 9.1691628
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1215286, 12.1177292
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1262207, 11.1217575
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4137344, 12.4077759
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7244797, 9.7175064
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9421310, 12.9468079
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2084122, 13.2115631
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4675188, 7.4686527
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3847427, 18.3927078
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7879715, 12.7943954
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4037094, 21.4151230
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8529701, 8.8560753
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0804138, 10.0770798
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6678467, 21.6700897
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6926155, 10.6918793
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5549793, 7.5553455
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2405128, 10.2413902
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8975487, 8.8994370
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6384506, 8.6432686
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6983013, 7.6988792
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1054592, 8.1037159
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3321438, 9.3333893
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4910316, 14.4983521
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0509396, 9.0488052
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6023674, 9.6025124
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5461903, 8.5527725
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6427917, 7.6432629
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1108093, 9.1086521
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6862259, 10.6848888
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5333939, 14.5288429
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6420021, 10.6338844
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6888275, 10.6875839
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6771774, 12.6772308
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0849991, 16.0836487
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1519165, 15.1466026
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1830139, 13.1807556
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1621742, 8.1549225
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2098694, 11.2043076
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5195332, 8.5155697

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0796168, upper bound: 5.0783962
time: 20.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0721293, upper bound: 5.0858396
time: 19.48 seconds

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

Time for backsubstitution: 2.19 seconds

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
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0757104, upper bound: 5.0823191
time: 15.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0682415, upper bound: 5.0897771
time: 17.18 seconds

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

Time for backsubstitution: 2.19 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0747937, upper bound: 5.0835956
time: 22.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0910666
time: 20.97 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 45.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0910666, upper bound: 5.0673420
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0835956, upper bound: 5.0747936
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0897771, upper bound: 5.0682415
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0823190, upper bound: 5.0757104
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0858396, upper bound: 5.0721293
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0783962, upper bound: 5.0796168
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0848997, upper bound: 5.0733404
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0774672, upper bound: 5.0808508
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0900202, upper bound: 5.0683890
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0825519, upper bound: 5.0758557
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0887125, upper bound: 5.0692829
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0812645, upper bound: 5.0767563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0847673, upper bound: 5.0731411
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0773290, upper bound: 5.0806484
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0838200, upper bound: 5.0743509
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0763922, upper bound: 5.0818602
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0818602, upper bound: 5.0763922
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0743509, upper bound: 5.0838200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0806484, upper bound: 5.0773290
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0731411, upper bound: 5.0847673
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0767563, upper bound: 5.0812645
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0692829, upper bound: 5.0887125
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0758557, upper bound: 5.0825519
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0900202
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0808508, upper bound: 5.0774672
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0733404, upper bound: 5.0848997
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0796168, upper bound: 5.0783962
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0721293, upper bound: 5.0858396
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0757104, upper bound: 5.0823191
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0682415, upper bound: 5.0897771
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0747937, upper bound: 5.0835956
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 45.50
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0910666

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4275894, 13.4369888
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3525620, 6.3586979
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5136871, 8.5211296
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0908318, 11.1012650
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1645775, 9.1779175
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1149292, 12.1239395
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1206322, 11.1254044
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4064865, 12.4154739
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7098198, 9.7285576
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9276428, 12.9258156
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2054520, 13.2031479
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4587345, 7.4518318
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3966599, 18.3820267
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7687759, 12.7613983
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4123459, 21.4002762
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8504753, 8.8549614
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0735931, 10.0764904
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6727905, 21.6646805
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6736488, 10.6715965
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5486450, 7.5469475
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2417831, 10.2413445
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8989296, 8.8955517
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6431293, 8.6385765
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6931400, 7.6896648
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0923004, 8.0910816
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3320065, 9.3280945
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4886589, 14.4822845
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0353909, 9.0392494
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6001358, 9.5968628
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5485916, 8.5401039
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6422348, 7.6348534
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1036530, 9.1044540
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6729546, 10.6765556
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5066605, 14.5170097
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6264915, 10.6400719
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6821632, 10.6864204
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6728821, 12.6767769
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0803680, 16.0861816
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1420898, 15.1541061
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1542435, 13.1642799
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1435509, 8.1571236
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2015572, 11.2109146
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5087376, 8.5147381

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0909730, upper bound: 5.0629395
time: 18.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0853373, upper bound: 5.0670328
time: 21.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4280624, 13.4365196
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3515968, 6.3596611
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5126762, 8.5221405
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0935097, 11.0985909
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1643524, 9.1781425
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1149902, 12.1238785
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1210365, 11.1250000
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4043655, 12.4175987
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7082024, 9.7301750
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9327011, 12.9207611
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2070084, 13.2015877
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4547253, 7.4558411
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4004745, 18.3782043
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7791824, 12.7509956
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4116821, 21.4009399
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8515015, 8.8539371
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0747681, 10.0753078
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6726685, 21.6648178
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6671333, 10.6781120
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5453758, 7.5502186
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2390327, 10.2440948
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8954315, 8.8990479
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6431293, 8.6385784
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6905308, 7.6922760
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0873032, 8.0960789
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3313580, 9.3287468
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4854622, 14.4854813
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0275326, 9.0471077
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5980186, 9.5989799
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5471458, 8.5415497
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6397209, 7.6373672
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.0994110, 9.1086979
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6767464, 10.6727619
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5142174, 14.5094528
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6321259, 10.6344376
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6883507, 10.6802292
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6770020, 12.6726608
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0836411, 16.0829086
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1432953, 15.1529007
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1619110, 13.1566124
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1460304, 8.1546402
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2027130, 11.2097588
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5119114, 8.5115643

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0835033, upper bound: 5.0703741
time: 26.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0778909, upper bound: 5.0744822
time: 15.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4281769, 13.4363327
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3528137, 6.3584061
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5150337, 8.5197792
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0908546, 11.1012154
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1669388, 9.1755562
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1159286, 12.1228714
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1207390, 11.1252899
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4073944, 12.4144669
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7126617, 9.7257156
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9262772, 12.9271812
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2042160, 13.2043800
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4583721, 7.4521751
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3937225, 18.3849640
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7657242, 12.7644577
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4107590, 21.4018555
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8519249, 8.8535137
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0724640, 10.0776119
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6701508, 21.6673279
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6752930, 10.6699524
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5493011, 7.5462914
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2431221, 10.2400055
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8992805, 8.8951988
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6432896, 8.6384163
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6932850, 7.6895199
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0931625, 8.0902157
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3312283, 9.3288727
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4902458, 14.4806976
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0386486, 9.0359917
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5999603, 9.5970383
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5485802, 8.5401154
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6411781, 7.6359119
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1043320, 9.1037731
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6729393, 10.6765690
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5064850, 14.5171509
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6264763, 10.6400909
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6816940, 10.6868286
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6738205, 12.6758308
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0814514, 16.0850983
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1448669, 15.1513290
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1544876, 13.1640320
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1449890, 8.1556816
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2032013, 11.2092743
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5086803, 8.5147953

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0896818, upper bound: 5.0638842
time: 20.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0840216, upper bound: 5.0679379
time: 22.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4286499, 13.4358597
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3518524, 6.3593693
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5140266, 8.5207901
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0935326, 11.0985413
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1667137, 9.1757812
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1159973, 12.1228104
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1211395, 11.1248856
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4052734, 12.4165916
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7110443, 9.7273331
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9313278, 12.9221268
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2057724, 13.2028236
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4543629, 7.4561863
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3975372, 18.3811417
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7761307, 12.7540512
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4100952, 21.4025269
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8529472, 8.8524895
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0736465, 10.0764332
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6700134, 21.6674652
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6687775, 10.6764679
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5460320, 7.5495605
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2403717, 10.2427559
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8957863, 8.8986950
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6432896, 8.6384182
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6906757, 7.6921310
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0881729, 8.0952110
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3305798, 9.3295288
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4870491, 14.4838943
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0307903, 9.0438480
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5978432, 9.5991554
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5471344, 8.5415611
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6386642, 7.6384239
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1000900, 9.1080170
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6767349, 10.6727753
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5140419, 14.5095940
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6321106, 10.6344528
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6878815, 10.6806412
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6779404, 12.6717148
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0847244, 16.0818253
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1460724, 15.1501236
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1621552, 13.1563683
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1474724, 8.1531982
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2043571, 11.2081146
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5118542, 8.5116215

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0822252, upper bound: 5.0713294
time: 16.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0765881, upper bound: 5.0754056
time: 16.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4301529, 13.4343567
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3537025, 6.3575153
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5164680, 8.5183487
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0919838, 11.1000900
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1688004, 9.1736946
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1174698, 12.1213379
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1205826, 11.1254463
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4090118, 12.4128494
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7165108, 9.7218666
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9265366, 12.9269180
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2051010, 13.2034988
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4578457, 7.4527016
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3914108, 18.3872681
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7630997, 12.7670784
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4123535, 21.4002686
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8537598, 8.8516788
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0727844, 10.0772991
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6699219, 21.6675568
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6755142, 10.6697311
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5496140, 7.5459785
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2433777, 10.2397499
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8999329, 8.8945465
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6432667, 8.6384430
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6929951, 7.6898098
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0933151, 8.0900631
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3309994, 9.3291016
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4907265, 14.4802094
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0401821, 9.0344582
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5996284, 9.5973663
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5483627, 8.5403366
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6400375, 7.6370544
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1050949, 9.1030121
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6721878, 10.6773205
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5057640, 14.5178719
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6264038, 10.6401596
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6805420, 10.6879807
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6727982, 12.6768570
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0809631, 16.0855865
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1448364, 15.1513596
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1542969, 13.1642227
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1454697, 8.1552010
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2028770, 11.2095947
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5081692, 8.5153065

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0855282, upper bound: 5.0664670
time: 18.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0814143, upper bound: 5.0720376
time: 22.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4306259, 13.4338875
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3527412, 6.3584785
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5154610, 8.5193596
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0946541, 11.0974159
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1685715, 9.1739235
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1175308, 12.1212692
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1209869, 11.1250458
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4068909, 12.4149704
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7148933, 9.7234840
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9315948, 12.9218636
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2066574, 13.2019424
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4538364, 7.4567127
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3952332, 18.3834534
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7735062, 12.7566719
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4116821, 21.4009323
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8547859, 8.8506527
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0739594, 10.0761166
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6697845, 21.6676941
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6689987, 10.6762466
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5463448, 7.5492496
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2406273, 10.2425003
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8964348, 8.8980427
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6432590, 8.6384449
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6903858, 7.6924191
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0883255, 8.0950603
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3303509, 9.3297577
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4875298, 14.4834137
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0323238, 9.0423164
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5975113, 9.5994873
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5469170, 8.5417824
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6375237, 7.6395664
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1008530, 9.1072540
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6759834, 10.6735268
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5133209, 14.5103149
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6320381, 10.6345253
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6867332, 10.6817894
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6769104, 12.6727409
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0842361, 16.0823135
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1460419, 15.1501541
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1619644, 13.1565552
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1479530, 8.1527195
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2040329, 11.2084389
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5113430, 8.5121326

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0780872, upper bound: 5.0739193
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0739907, upper bound: 5.0795231
time: 17.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4308090, 13.4337692
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3539963, 6.3572636
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5178185, 8.5169983
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0920296, 11.1000633
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1711617, 9.1713333
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1185379, 12.1203308
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1206970, 11.1253433
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4100189, 12.4119415
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7193565, 9.7190247
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9251785, 12.9282837
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2038651, 13.2047348
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4575024, 7.4530640
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3884735, 18.3902054
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7600479, 12.7701340
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4107742, 21.4018555
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8552094, 8.8502293
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0716553, 10.0784206
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6672668, 21.6702042
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6771545, 10.6680908
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5502701, 7.5453224
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2447166, 10.2384109
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9002838, 8.8941936
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6434269, 8.6382828
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6931400, 7.6896667
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0941849, 8.0891972
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3302212, 9.3298798
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4923210, 14.4786224
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0434399, 9.0312004
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5994530, 9.5975418
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5483475, 8.5403481
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6389771, 7.6381111
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1057777, 9.1023293
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6721764, 10.6773338
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5056229, 14.5180473
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6263885, 10.6401749
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6801338, 10.6884460
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6737442, 12.6759186
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0820465, 16.0845032
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1476135, 15.1485786
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1545486, 13.1639748
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1469116, 8.1537609
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2045212, 11.2079506
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5081081, 8.5153675

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0845840, upper bound: 5.0677373
time: 21.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0804312, upper bound: 5.0732498
time: 13.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4312820, 13.4333000
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3530350, 6.3582249
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5168076, 8.5180092
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0947075, 11.0973892
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1709328, 9.1715622
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1185989, 12.1202698
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1211014, 11.1249390
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4078979, 12.4140625
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7177353, 9.7206421
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9302292, 12.9232292
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2054214, 13.2031784
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4534893, 7.4570751
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3922958, 18.3863907
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7704544, 12.7597275
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4101028, 21.4025192
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8562355, 8.8492050
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0728378, 10.0772419
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6671448, 21.6703415
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6706390, 10.6746063
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5470009, 7.5485916
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2419624, 10.2411613
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8967896, 8.8976898
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6434193, 8.6382828
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6905308, 7.6922779
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0891876, 8.0941925
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3295650, 9.3305359
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4891167, 14.4818268
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0355816, 9.0390568
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5973358, 9.5996628
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5469055, 8.5417938
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6364670, 7.6406250
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1015358, 9.1065712
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6759682, 10.6735420
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5131798, 14.5104904
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6320229, 10.6345444
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6863213, 10.6822586
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6778564, 12.6718025
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0853195, 16.0812302
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1488190, 15.1473732
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1622162, 13.1563072
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1493950, 8.1512794
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2056770, 11.2067947
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5112820, 8.5121937

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0771538, upper bound: 5.0752085
time: 21.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0730214, upper bound: 5.0807602
time: 25.79 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0899262, upper bound: 5.0640107
time: 15.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0842643, upper bound: 5.0680835
time: 20.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4305344, 13.4340439
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3548546, 6.3564072
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5154839, 8.5193367
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0979652, 11.0941315
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1687775, 9.1737213
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1179962, 12.1208725
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1227074, 11.1233292
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4095535, 12.4124107
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7154350, 9.7229424
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9278946, 12.9255676
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2044754, 13.2041168
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4548512, 7.4557171
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3909073, 18.3877792
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7723770, 12.7578049
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4018173, 21.4108047
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8503380, 8.8551025
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0766907, 10.0733852
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6703949, 21.6670837
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6680908, 10.6771545
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5454254, 7.5501671
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2383270, 10.2448006
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8936806, 8.9007988
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6393299, 8.6423779
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6907482, 7.6920567
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0905914, 8.0927868
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3317776, 9.3283234
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4801369, 14.4908066
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0312710, 9.0433712
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5980339, 9.5989647
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5415573, 8.5471420
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6392517, 7.6378365
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1027718, 9.1053352
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6767502, 10.6727581
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5176735, 14.5060005
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6387215, 10.6278419
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6883583, 10.6802216
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6764221, 12.6732407
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0840225, 16.0825272
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1474152, 15.1487808
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1639481, 13.1545715
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1527672, 8.1479034
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2064362, 11.2060356
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5147991, 8.5086784

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0824582, upper bound: 5.0714510
time: 18.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0768234, upper bound: 5.0755455
time: 21.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4306564, 13.4338570
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3560677, 6.3551521
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5178413, 8.5169754
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.0953178, 11.0967560
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.1713638, 9.1711311
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1189346, 12.1198730
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1224098, 11.1236191
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4125824, 12.4092827
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7198944, 9.7184830
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9214706, 12.9319878
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2016830, 13.2069130
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4584980, 7.4520512
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.3841476, 18.3945389
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.7589111, 12.7712669
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4008942, 21.4117203
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8507576, 8.8546791
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0743866, 10.0756893
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6678772, 21.6696014
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6762505, 10.6689949
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5493546, 7.5462379
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2424126, 10.2407112
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8975296, 8.8969498
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6394901, 8.6422157
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6935024, 7.6893024
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.0964584, 8.0869236
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3316479, 9.3284531
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.4849205, 14.4860229
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0423870, 9.0322552
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.5999718, 9.5970230
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5429916, 8.5457077
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6407089, 7.6363811
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1076965, 9.1004124
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6729431, 10.6765652
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5099373, 14.5136986
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6330719, 10.6334915
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6817017, 10.6868210
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6732407, 12.6764107
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0818329, 16.0847168
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1489868, 15.1472092
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1565323, 13.1619911
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1517258, 8.1489449
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2069244, 11.2055473
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5115643, 8.5119114

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0886169, upper bound: 5.0649485
time: 20.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0829378, upper bound: 5.0689832
time: 27.15 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 49.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0909730, upper bound: 5.0629395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0853373, upper bound: 5.0670328
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0835033, upper bound: 5.0703741
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0778909, upper bound: 5.0744822
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0896818, upper bound: 5.0638842
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0840216, upper bound: 5.0679379
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0822252, upper bound: 5.0713294
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0765881, upper bound: 5.0754056
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0855282, upper bound: 5.0664670
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0814143, upper bound: 5.0720376
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0780872, upper bound: 5.0739193
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0739907, upper bound: 5.0795231
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0845840, upper bound: 5.0677373
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0804312, upper bound: 5.0732498
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0771538, upper bound: 5.0752085
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0730214, upper bound: 5.0807602
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0899262, upper bound: 5.0640107
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0842643, upper bound: 5.0680835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0824582, upper bound: 5.0714510
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0768234, upper bound: 5.0755455
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0886169, upper bound: 5.0649485
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 49.93
Output dim: 11, lower bound: -5.0829378, upper bound: 5.0689832
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0812645, upper bound: 5.0767563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0847673, upper bound: 5.0731411
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0773290, upper bound: 5.0806484
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0838200, upper bound: 5.0743509
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0763922, upper bound: 5.0818602
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0818602, upper bound: 5.0763922
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0743509, upper bound: 5.0838200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0806484, upper bound: 5.0773290
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0731411, upper bound: 5.0847673
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0767563, upper bound: 5.0812645
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0692829, upper bound: 5.0887125
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0758557, upper bound: 5.0825519
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0900202
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0808508, upper bound: 5.0774672
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0733404, upper bound: 5.0848997
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0796168, upper bound: 5.0783962
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0721293, upper bound: 5.0858396
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0757104, upper bound: 5.0823191
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0682415, upper bound: 5.0897771
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0747937, upper bound: 5.0835956
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 49.93
Output dim: 11, lower bound: -5.0673420, upper bound: 5.0910666

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 30.30 + 1771.46 = 1801.76 seconds
