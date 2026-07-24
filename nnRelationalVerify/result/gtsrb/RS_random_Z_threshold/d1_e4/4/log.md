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
execution time: IAR + RelationalAnalysis = 2.76 + 26.28 = 29.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -5.0948664, upper bound: 5.0948664

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1632

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0944556, upper bound: 5.0946629
time: 20.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0946629, upper bound: 5.0944556
time: 7.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 28.93 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 28.93
Output dim: 11, lower bound: -5.0944556, upper bound: 5.0946629
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 28.93
Output dim: 11, lower bound: -5.0946629, upper bound: 5.0944556

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4259033, 13.4268417
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3950081, 6.3957195
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5666847, 8.5667038
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1578598, 11.1585464
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2555161, 9.2559280
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1758270, 12.1760178
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1567039, 11.1559715
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4873123, 12.4881096
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8317871, 9.8324471
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9245758, 12.9251480
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2186661, 13.2188110
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4874468, 7.4874687
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.5017014, 18.5014572
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8731651, 12.8740005
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4890442, 21.4898376
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8924751, 8.8924980
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1102295, 10.1106739
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7154999, 21.7162170
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6780891, 10.6774750
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5611935, 7.5609016
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2531548, 10.2529182
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9095650, 8.9095631
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6767979, 8.6766205
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7109718, 7.7108688
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1389694, 8.1388969
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3570900, 9.3574600
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5628319, 14.5617218
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1024933, 9.1021652
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6109962, 9.6105919
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6023884, 8.6019592
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6532707, 7.6532745
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1468201, 9.1468048
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6891136, 10.6887894
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5446854, 14.5446815
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6615486, 10.6610184
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7014542, 10.7010689
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6873322, 12.6867638
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0834579, 16.0828247
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1168442, 15.1166039
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1905556, 13.1907578
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1941719, 8.1936378
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2379532, 11.2369232
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5127239, 8.5126247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1322

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0943243, upper bound: 5.0944246
time: 31.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0942173, upper bound: 5.0945316
time: 21.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4268417, 13.4258995
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3957176, 6.3950100
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5667038, 8.5666847
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1585464, 11.1578636
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2559280, 9.2555161
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1760178, 12.1758270
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1559715, 11.1567039
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4881134, 12.4873161
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8324471, 9.8317871
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9251480, 12.9245796
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2188110, 13.2186623
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4874697, 7.4874477
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.5014572, 18.5016937
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8740044, 12.8731689
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4898376, 21.4890442
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8924980, 8.8924751
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1106720, 10.1102314
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7162170, 21.7154922
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6774750, 10.6780891
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5608997, 7.5611935
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2529182, 10.2531548
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9095612, 8.9095650
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6766186, 8.6767960
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7108688, 7.7109737
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1388969, 8.1389713
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3574600, 9.3570900
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5617256, 14.5628357
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1021652, 9.1024933
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6105919, 9.6109962
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6019611, 8.6023903
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6532707, 7.6532726
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1468048, 9.1468201
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6887894, 10.6891136
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5446815, 14.5446854
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6610184, 10.6615524
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7010689, 10.7014542
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6867599, 12.6873283
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0828323, 16.0834579
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1166077, 15.1168442
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1907539, 13.1905518
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1936378, 8.1941719
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2369232, 11.2379532
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5126247, 8.5127239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 705

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0928031, upper bound: 5.0944374
time: 19.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0946447, upper bound: 5.0925958
time: 18.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 40.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 40.41
Output dim: 11, lower bound: -5.0943243, upper bound: 5.0944246
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 40.41
Output dim: 11, lower bound: -5.0942173, upper bound: 5.0945316
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 40.41
Output dim: 11, lower bound: -5.0928031, upper bound: 5.0944374
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 40.41
Output dim: 11, lower bound: -5.0946447, upper bound: 5.0925958

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4248581, 13.4254913
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3945923, 6.3951836
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5666199, 8.5666199
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1577377, 11.1584740
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2551117, 9.2556152
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1752243, 12.1755295
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1564064, 11.1556511
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4870148, 12.4878578
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8315353, 9.8320389
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9243164, 12.9248123
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2185822, 13.2186890
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4867325, 7.4867172
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.5016251, 18.5013962
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8725090, 12.8734741
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4888077, 21.4894714
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8921585, 8.8921089
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1096573, 10.1099567
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7151947, 21.7159653
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6779404, 10.6773453
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5610332, 7.5607853
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2526741, 10.2525635
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9090767, 8.9093056
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6761837, 8.6762314
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7104168, 7.7101460
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1385345, 8.1384792
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3570251, 9.3574142
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5623436, 14.5611801
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1023159, 9.1019497
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6106911, 9.6101952
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6022701, 8.6018219
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6529369, 7.6529598
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1464348, 9.1465378
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6890373, 10.6887150
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5441360, 14.5441284
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6611710, 10.6606140
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7009583, 10.7006683
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6866379, 12.6861877
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0829239, 16.0820312
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1163635, 15.1162033
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1903992, 13.1906281
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1926651, 8.1917648
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2373581, 11.2361870
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5111809, 8.5106983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 641

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0941866, upper bound: 5.0869263
time: 22.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0868272, upper bound: 5.0942870
time: 24.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4245529, 13.4257927
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3944740, 6.3953018
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5666008, 8.5666389
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1577911, 11.1584206
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2552071, 9.2555199
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1753464, 12.1754074
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1563873, 11.1556740
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4870605, 12.4878159
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8313789, 9.8321915
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9242401, 12.9248886
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2185364, 13.2187386
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4866982, 7.4867535
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.5016327, 18.5013885
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8726463, 12.8733368
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4886780, 21.4896011
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8920822, 8.8921814
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1095123, 10.1101017
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7152557, 21.7159195
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6779633, 10.6773262
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5610790, 7.5607395
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2528000, 10.2524376
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9093094, 8.9090729
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6764050, 8.6760082
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7102489, 7.7103119
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1385574, 8.1384563
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3570442, 9.3573952
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5622826, 14.5612411
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1022778, 9.1019878
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6105995, 9.6102867
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6022511, 8.6018448
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6529598, 7.6529388
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1465530, 9.1464233
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6890411, 10.6887131
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5441322, 14.5441322
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6611481, 10.6606407
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7010536, 10.7005730
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6867599, 12.6860733
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0826569, 16.0822983
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1164474, 15.1161194
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1904221, 13.1906013
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1922989, 8.1921310
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2372169, 11.2363281
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5107994, 8.5110798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1285

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1418

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0940351, upper bound: 5.0918475
time: 18.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0915390, upper bound: 5.0943494
time: 24.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4084167, 13.4052467
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3859653, 6.3839531
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5597115, 8.5588036
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1575317, 11.1567154
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2495575, 9.2483749
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1703033, 12.1693192
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1499596, 11.1514359
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4823227, 12.4806175
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8050117, 9.8006287
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9153824, 12.9135704
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2059021, 13.2039185
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4883499, 7.4883633
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4900208, 18.4916840
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8752098, 12.8759270
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4565735, 21.4513168
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8759041, 8.8736534
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1007805, 10.0989361
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6985168, 21.6954269
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6775589, 10.6781883
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5605793, 7.5609207
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2486267, 10.2493935
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9043198, 8.9048996
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6723366, 8.6729126
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7095356, 7.7098484
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1448936, 8.1448421
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3591156, 9.3582954
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5664368, 14.5675201
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1050186, 9.1057320
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6106873, 9.6111298
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6032410, 8.6032104
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6592407, 7.6585102
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1425819, 9.1431580
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6654053, 10.6686363
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5267563, 14.5289879
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6425095, 10.6453400
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6746635, 10.6783295
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6594849, 12.6634407
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0836792, 16.0844650
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0886917, 15.0924034
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1707954, 13.1730690
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.2015991, 8.2027931
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2261238, 11.2284927
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5179291, 8.5178642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 545

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0923870, upper bound: 5.0896400
time: 18.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0893750, upper bound: 5.0944371
time: 19.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4061890, 13.4074707
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3846607, 6.3852539
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5588264, 8.5596924
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1574020, 11.1568489
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2487831, 9.2491493
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1695099, 12.1701202
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1507034, 11.1506882
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4814148, 12.4815292
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8012848, 9.8043556
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9141312, 12.9148140
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2040710, 13.2057495
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4883842, 7.4883270
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4914398, 18.4902573
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8767586, 12.8743744
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4521103, 21.4557800
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8736763, 8.8758812
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0993767, 10.1003361
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6961670, 21.6977997
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6775703, 10.6781731
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5606289, 7.5608749
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2491570, 10.2488670
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9048958, 8.9043236
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6727371, 8.6725121
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7097416, 7.7096405
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1447678, 8.1449699
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3586655, 9.3587456
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5664215, 14.5675507
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1054077, 9.1053429
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6107254, 9.6110878
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6027832, 8.6036682
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6585121, 7.6592388
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1431427, 9.1425953
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6683083, 10.6657314
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5289803, 14.5267601
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6448097, 10.6430397
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6779442, 10.6750526
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6628723, 12.6600571
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0838394, 16.0843048
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0921631, 15.0889359
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1732750, 13.1705894
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.2022591, 8.2021332
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2274628, 11.2271538
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5177650, 8.5180283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1770

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0887546, upper bound: 5.0920328
time: 64.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0940826, upper bound: 5.0887546
time: 16.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 83.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 83.70
Output dim: 11, lower bound: -5.0941866, upper bound: 5.0869263
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 83.70
Output dim: 11, lower bound: -5.0868272, upper bound: 5.0942870
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 83.70
Output dim: 11, lower bound: -5.0940351, upper bound: 5.0918475
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 83.70
Output dim: 11, lower bound: -5.0915390, upper bound: 5.0943494
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 83.70
Output dim: 11, lower bound: -5.0923870, upper bound: 5.0896400
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 83.70
Output dim: 11, lower bound: -5.0893750, upper bound: 5.0944371
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 83.70
Output dim: 11, lower bound: -5.0887546, upper bound: 5.0920328
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 83.70
Output dim: 11, lower bound: -5.0940826, upper bound: 5.0887546

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4237671, 13.4267921
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3926239, 6.3947182
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5617142, 8.5631618
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1540298, 11.1558189
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2469978, 9.2498207
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1693954, 12.1713638
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1539955, 11.1535416
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4812775, 12.4837570
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8231277, 9.8264771
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9207153, 12.9225693
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2165985, 13.2173424
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4831238, 7.4816074
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.5004272, 18.4990768
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8692627, 12.8688889
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4887238, 21.4898300
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8919640, 8.8921776
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1067123, 10.1088295
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7148514, 21.7156372
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6777573, 10.6771507
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5610104, 7.5607662
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2522354, 10.2519531
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9086876, 8.9092007
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6746483, 8.6739998
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7080135, 7.7067356
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1351910, 8.1341610
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3527546, 9.3513832
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5619659, 14.5607147
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1018505, 9.1014175
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6092224, 9.6083717
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6003094, 8.5990601
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6502724, 7.6491737
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1448784, 9.1443577
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6889954, 10.6886234
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5439034, 14.5439110
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6608353, 10.6603737
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7002182, 10.6997337
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6861763, 12.6844330
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0821609, 16.0814362
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1155014, 15.1150246
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1905212, 13.1899490
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1909828, 8.1905708
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2360916, 11.2352943
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5110970, 8.5105534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 956

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1648

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0933013, upper bound: 5.0865422
time: 21.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0937884, upper bound: 5.0858747
time: 23.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4261551, 13.4244080
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3941269, 6.3932171
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5631638, 8.5617161
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1550827, 11.1547699
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2493134, 9.2475052
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1710587, 12.1697083
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1542931, 11.1532440
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4829178, 12.4821205
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8259735, 9.8236313
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9220734, 12.9212074
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2172394, 13.2166939
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4816208, 7.4831104
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4993057, 18.5002060
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8679199, 12.8702278
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4891663, 21.4893875
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8922272, 8.8919163
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1085358, 10.1070061
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7148819, 21.7156067
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6777458, 10.6771622
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5610142, 7.5607643
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2520676, 10.2521210
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9089699, 8.9089184
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6739540, 8.6746960
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7070103, 7.7077389
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1342144, 8.1351395
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3509960, 9.3531456
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5618744, 14.5607986
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1017818, 9.1014862
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6088638, 9.6087265
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5995083, 8.5998611
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6491508, 7.6502991
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1442566, 9.1449814
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6889458, 10.6886749
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5439224, 14.5438919
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6609306, 10.6602783
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7000237, 10.6999245
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6848793, 12.6857300
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0823364, 16.0812683
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1151810, 15.1153488
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1897125, 13.1907539
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1914711, 8.1900826
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2364655, 11.2349205
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5110359, 8.5106163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1447

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0856739, upper bound: 5.0909387
time: 18.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0835783, upper bound: 5.0931319
time: 15.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4233170, 13.4247894
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3927307, 6.3937988
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5668945, 8.5669422
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1548538, 11.1558304
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2540779, 9.2544899
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1733704, 12.1736450
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1571007, 11.1564827
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4863129, 12.4871368
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8257027, 9.8271904
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9253235, 12.9258194
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2175255, 13.2178307
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4866962, 7.4867516
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4980087, 18.4973145
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8693161, 12.8696098
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4869385, 21.4875565
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8891411, 8.8893986
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1095924, 10.1101894
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7129364, 21.7133789
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6777687, 10.6771202
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5613365, 7.5609360
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2527008, 10.2523346
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9095345, 8.9092827
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6764030, 8.6760006
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7100143, 7.7100563
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1363640, 8.1364307
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3554878, 9.3560524
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5619354, 14.5608368
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1015320, 9.1014137
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6105957, 9.6102791
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6020050, 8.6015358
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6524353, 7.6523476
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1456833, 9.1455917
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6876259, 10.6873779
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5437927, 14.5440636
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6595306, 10.6595879
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7010231, 10.7006912
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6866646, 12.6859818
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0837326, 16.0835571
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1182861, 15.1182976
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1901093, 13.1903839
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1935387, 8.1937160
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2373772, 11.2365417
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5123501, 8.5132160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1465

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0940098, upper bound: 5.0918003
time: 23.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0914918, upper bound: 5.0918104
time: 25.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4235458, 13.4245567
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3929672, 6.3935585
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5668983, 8.5669346
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1552048, 11.1554794
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2541809, 9.2543869
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1735764, 12.1734390
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1571960, 11.1563911
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4863815, 12.4870682
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8263741, 9.8265152
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9251785, 12.9259682
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2176323, 13.2177238
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4866962, 7.4867554
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4975662, 18.4977646
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8689117, 12.8700104
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4866257, 21.4878693
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8893051, 8.8892365
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1096001, 10.1101856
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7126923, 21.7136078
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6777573, 10.6771317
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5612755, 7.5609989
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2526932, 10.2523422
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9095154, 8.9092999
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6763992, 8.6760063
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7099915, 7.7100754
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1365242, 8.1362629
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3556976, 9.3558388
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5618820, 14.5608902
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1017036, 9.1012421
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6105919, 9.6102829
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6019402, 8.6015968
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6523666, 7.6524162
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1457214, 9.1455555
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6877060, 10.6872959
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5440598, 14.5437927
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6600914, 10.6590233
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7011757, 10.7005424
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6866646, 12.6859818
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0839157, 16.0833740
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1186218, 15.1179619
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1902084, 13.1902847
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1938820, 8.1933708
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2374344, 11.2364883
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5129375, 8.5126324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1712

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1760

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0892179, upper bound: 5.0943494
time: 20.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0915390, upper bound: 5.0920221
time: 26.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3669281, 13.3692284
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3659439, 6.3666286
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5436440, 8.5448189
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1354828, 11.1375389
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2323494, 9.2333260
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1478271, 12.1496124
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1249123, 11.1228371
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4587555, 12.4600334
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7572517, 9.7587090
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8900223, 12.8916206
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1748428, 13.1767159
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4835234, 7.4832287
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4707642, 18.4696884
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8791580, 12.8791428
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4186325, 21.4178238
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8566666, 8.8566284
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0818329, 10.0823402
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6658859, 21.6667709
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6650047, 10.6637993
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5513878, 7.5502415
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2388229, 10.2380104
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8924522, 8.8914490
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6616592, 8.6608925
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7105217, 7.7105312
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1349621, 8.1342163
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3571529, 9.3563347
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5596619, 14.5600739
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0901260, 9.0892181
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6086044, 9.6076775
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6016045, 8.6016121
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6567287, 7.6561069
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1283493, 9.1269684
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6413956, 10.6412182
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4979057, 14.4960442
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6202812, 10.6199570
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6412773, 10.6402054
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6181870, 12.6162796
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0676575, 16.0661469
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0452042, 15.0427399
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1378174, 13.1354141
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1973801, 8.1982002
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2030354, 11.2021332
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5207291, 8.5207806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 870

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1777

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0889051, upper bound: 5.0895301
time: 20.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0922771, upper bound: 5.0861632
time: 16.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3725128, 13.3637543
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3685913, 6.3639336
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5456314, 8.5427399
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1383438, 11.1346626
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2345772, 9.2311668
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1505432, 12.1468430
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1213570, 11.1263885
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4616623, 12.4570541
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7631874, 9.7528687
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8934326, 12.8882027
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1786957, 13.1728592
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4832144, 7.4835682
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4680328, 18.4724197
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8784256, 12.8799019
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4230270, 21.4133759
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8589935, 8.8544121
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0841904, 10.0799866
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6699448, 21.6627884
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6631699, 10.6656342
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5499001, 7.5517292
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2372475, 10.2395439
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8908691, 8.8930969
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6603165, 8.6622868
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7102165, 7.7108383
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1342716, 8.1349640
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3571529, 9.3563576
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5589905, 14.5608521
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0885010, 9.0909138
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6072311, 9.6090546
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6016388, 8.6016502
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6568394, 7.6560192
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1263924, 9.1289291
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6379890, 10.6446247
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4938126, 14.5001373
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6171265, 10.6231079
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6365395, 10.6449432
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6123276, 12.6221428
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0653534, 16.0684586
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0390320, 15.0489120
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1331406, 13.1400909
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1970062, 8.1985970
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1997623, 11.2054100
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5208168, 8.5206661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1770

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0857145, upper bound: 5.0938742
time: 17.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0890125, upper bound: 5.0905971
time: 15.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4046135, 13.4040337
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3837318, 6.3830795
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5569382, 8.5558167
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1565590, 11.1553764
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2460480, 9.2433853
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1681061, 12.1675949
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1505699, 11.1506233
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4790955, 12.4767570
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7975464, 9.7962570
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9139709, 12.9158020
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2039642, 13.2061691
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4874611, 7.4876652
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4851379, 18.4872055
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8684425, 12.8709717
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4520569, 21.4557037
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8724022, 8.8736649
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0990944, 10.1007862
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6932068, 21.6953125
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6785278, 10.6779861
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5601883, 7.5597839
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2482300, 10.2467499
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9043846, 8.9031372
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6726379, 8.6723595
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7096500, 7.7093925
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1448250, 8.1449203
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3552780, 9.3569870
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5663414, 14.5674896
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1035004, 9.0997353
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6106148, 9.6109962
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6019402, 8.6029587
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6560402, 7.6579399
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1423569, 9.1406689
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6676598, 10.6646919
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5285034, 14.5264549
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6447144, 10.6428337
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6754570, 10.6739235
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6610413, 12.6591034
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0836182, 16.0841293
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0928650, 15.0887833
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1723518, 13.1700020
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.2016678, 8.2010498
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2263260, 11.2257538
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5177193, 8.5179729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1482

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0886171, upper bound: 5.0910364
time: 21.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0897967, upper bound: 5.0917007
time: 20.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4027519, 13.4059029
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3824883, 6.3843231
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5549545, 8.5578041
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1559258, 11.1560097
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2430191, 9.2464142
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1669846, 12.1687164
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1506386, 11.1505585
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4766388, 12.4792137
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7931862, 9.8006172
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9151230, 12.9146461
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2044907, 13.2056465
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4877205, 7.4874039
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4883881, 18.4839554
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8733559, 12.8660545
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4520264, 21.4557266
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8714638, 8.8746071
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0998268, 10.1000538
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6936646, 21.6948547
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6773834, 10.6791306
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5595360, 7.5604324
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2470436, 10.2479362
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9037094, 8.9038124
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6725845, 8.6724148
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7094936, 7.7095490
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1447182, 8.1450310
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3569107, 9.3553581
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5663643, 14.5674744
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0998001, 9.1034412
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6106377, 9.6109772
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6020737, 8.6028252
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6572113, 7.6567669
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1412201, 9.1418095
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6672707, 10.6650810
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5286751, 14.5262833
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6445999, 10.6429482
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6768188, 10.6725655
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6619186, 12.6582222
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0836639, 16.0840836
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0920105, 15.0896416
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1726875, 13.1696701
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.2011757, 8.2015419
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2260666, 11.2260170
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5177078, 8.5179844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 529

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0940565, upper bound: 5.0865483
time: 27.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0918869, upper bound: 5.0887284
time: 17.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 46.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0933013, upper bound: 5.0865422
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0937884, upper bound: 5.0858747
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0856739, upper bound: 5.0909387
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0835783, upper bound: 5.0931319
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0940098, upper bound: 5.0918003
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0914918, upper bound: 5.0918104
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0892179, upper bound: 5.0943494
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0915390, upper bound: 5.0920221
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0889051, upper bound: 5.0895301
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0922771, upper bound: 5.0861632
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0857145, upper bound: 5.0938742
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0890125, upper bound: 5.0905971
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0886171, upper bound: 5.0910364
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0897967, upper bound: 5.0917007
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0940565, upper bound: 5.0865483
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.76
Output dim: 11, lower bound: -5.0918869, upper bound: 5.0887284

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4114990, 13.4153862
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3855667, 6.3881073
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5618324, 8.5633144
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1504593, 11.1531181
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2475052, 9.2504387
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1709442, 12.1731339
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1418190, 11.1396255
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4772873, 12.4804497
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8138046, 9.8177223
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9104004, 12.9130363
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2116852, 13.2128067
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4830408, 7.4815369
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4958572, 18.4941940
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8571472, 12.8582916
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4794312, 21.4813919
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8906364, 8.8908768
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1010513, 10.1034203
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7054749, 21.7073669
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6695862, 10.6678123
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5593719, 7.5584335
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2472267, 10.2464943
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9072609, 8.9075947
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6712685, 8.6704273
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7095871, 7.7082195
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1379890, 8.1365662
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3474674, 9.3467560
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5548630, 14.5522308
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0943947, 9.0930042
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6132774, 9.6116409
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6001930, 8.5983505
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6503448, 7.6492252
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1443787, 9.1438351
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6835899, 10.6828194
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5414047, 14.5411491
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6530418, 10.6514664
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6944580, 10.6930771
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6781464, 12.6758080
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0844498, 16.0827789
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1077423, 15.1065636
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1894188, 13.1891594
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1854935, 8.1842957
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2244129, 11.2219429
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5111084, 8.5105782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1418

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0931192, upper bound: 5.0839136
time: 17.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0906360, upper bound: 5.0863601
time: 20.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4123535, 13.4145317
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3860168, 6.3876591
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5618668, 8.5632801
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1513596, 11.1522369
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2476196, 9.2503242
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1711426, 12.1729126
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1400795, 11.1413612
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4779358, 12.4797592
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8143730, 9.8171539
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9111786, 12.9122543
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2120438, 13.2124329
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4830523, 7.4815254
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4955444, 18.4944992
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8586655, 12.8567772
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4802628, 21.4805374
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8906631, 8.8908520
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1013031, 10.1031685
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7065735, 21.7062759
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6684189, 10.6689796
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5586777, 7.5591278
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2467728, 10.2469444
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9070816, 8.9077721
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6710777, 8.6706181
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7094955, 7.7082901
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1375999, 8.1369133
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3481236, 9.3460922
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5534744, 14.5536118
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0934372, 9.0939407
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6124916, 9.6124306
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5995979, 8.5989456
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6503258, 7.6492405
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1443596, 9.1438541
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6831932, 10.6832161
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5411415, 14.5414047
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6519279, 10.6525803
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6935577, 10.6939735
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6775513, 12.6764030
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0835037, 16.0837250
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1070404, 15.1072693
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1897316, 13.1888466
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1847076, 8.1850815
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2227421, 11.2236137
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5111084, 8.5105629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1681

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0937633, upper bound: 5.0858413
time: 20.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0937548, upper bound: 5.0858497
time: 20.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4251709, 13.4224968
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3939400, 6.3930168
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5629807, 8.5615883
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1535721, 11.1528282
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2434273, 9.2425728
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1700745, 12.1683960
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1527328, 11.1516342
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4826279, 12.4818726
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8250961, 9.8229904
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9206085, 12.9192734
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2150269, 13.2138062
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4816017, 7.4831009
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4971313, 18.4971237
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8666267, 12.8682327
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4887314, 21.4885941
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8896866, 8.8901253
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1081581, 10.1063080
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7141342, 21.7144318
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6750908, 10.6753464
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5563869, 7.5571537
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2516365, 10.2520676
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9073334, 8.9078293
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6714172, 8.6728020
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7066116, 7.7074356
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1292324, 8.1315174
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3493748, 9.3524361
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5588531, 14.5584641
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0975285, 9.0984535
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6078644, 9.6080933
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5987873, 8.5992432
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6486244, 7.6500187
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1389008, 9.1409969
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6855965, 10.6848488
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5423622, 14.5431633
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6600151, 10.6596107
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6981697, 10.6986427
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6843605, 12.6853447
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0800705, 16.0796738
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1150894, 15.1152802
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1847687, 13.1873550
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1913795, 8.1900349
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2348766, 11.2339516
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5040970, 8.5025520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1351

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1378

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0844533, upper bound: 5.0909112
time: 21.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0856447, upper bound: 5.0897266
time: 18.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4242401, 13.4234276
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3939285, 6.3930283
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5630341, 8.5615349
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1531448, 11.1532669
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2443810, 9.2416191
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1697464, 12.1687164
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1526833, 11.1516838
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4826660, 12.4818306
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8253365, 9.8227501
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9201431, 12.9197388
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2143478, 13.2144814
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4816132, 7.4830875
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4962234, 18.4980316
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8659248, 12.8689346
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4883804, 21.4889450
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8904343, 8.8893776
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1078339, 10.1066322
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7136765, 21.7148819
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6759300, 10.6745071
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5574055, 7.5561371
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2520142, 10.2516937
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9078827, 8.9072819
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6720581, 8.6721592
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7067032, 7.7073441
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1305904, 8.1301594
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3502865, 9.3515282
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5595398, 14.5577774
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0987492, 9.0972328
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6082344, 9.6077271
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5988941, 8.5991402
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6488686, 7.6497726
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1402702, 9.1396294
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6851196, 10.6853256
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5431900, 14.5423393
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6602631, 10.6593590
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6987381, 10.6980743
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6844978, 12.6852074
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0807419, 16.0790100
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1151123, 15.1152573
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1863174, 13.1858063
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1914215, 8.1899948
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2354946, 11.2333298
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5029716, 8.5036755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1514

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0815611, upper bound: 5.0930052
time: 20.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0815611, upper bound: 5.0911227
time: 24.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3988190, 13.4032707
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3831062, 6.3853970
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5642128, 8.5645466
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1535492, 11.1546249
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2537842, 9.2542801
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1697159, 12.1702614
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1295280, 11.1249695
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4836197, 12.4847908
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7977066, 9.8026962
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9086990, 12.9112701
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1938171, 13.1970901
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4856110, 7.4857044
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4844513, 18.4818192
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8605232, 12.8586617
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4660645, 21.4692917
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8601685, 8.8640499
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1082344, 10.1089993
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7059937, 21.7073135
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6768112, 10.6754265
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5650616, 7.5639172
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2501640, 10.2493935
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9057274, 8.9045601
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6756554, 8.6753178
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7031097, 7.7037888
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1365166, 8.1365929
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3483315, 9.3498230
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5545807, 14.5543060
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1014137, 9.1011963
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6091461, 9.6088638
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5933971, 8.5939980
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6523170, 7.6525822
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1399994, 9.1382065
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6657906, 10.6624260
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5189133, 14.5156288
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6471939, 10.6455460
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6774597, 10.6737633
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6609497, 12.6565895
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0848007, 16.0840912
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0889359, 15.0847473
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1553497, 13.1506615
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1884384, 8.1881218
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2209206, 11.2177353
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5163021, 8.5173759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1415

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1465

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0935912, upper bound: 5.0861309
time: 12.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0883924, upper bound: 5.0913737
time: 19.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4017944, 13.4002953
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3843269, 6.3841763
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5644989, 8.5642605
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1536407, 11.1545296
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2538681, 9.2541962
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1699905, 12.1699944
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1255875, 11.1289101
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4839706, 12.4844437
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8012085, 9.7991943
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9107742, 12.9091949
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1967850, 13.1941261
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4856491, 7.4856663
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4825134, 18.4837570
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8583641, 12.8608170
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4686737, 21.4666824
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8637924, 8.8604279
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1084023, 10.1088314
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7068634, 21.7064514
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6760712, 10.6761665
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5643177, 7.5646610
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2497635, 10.2497978
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9048119, 8.9054737
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6757202, 8.6752510
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7037468, 7.7031555
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1365242, 8.1365871
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3492546, 9.3488998
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5554123, 14.5534744
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1013145, 9.1012974
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6091805, 9.6088333
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5944691, 8.5929222
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6526680, 7.6522350
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1382980, 9.1399040
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6626701, 10.6655464
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5153618, 14.5191803
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6454887, 10.6472473
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6740952, 10.6771278
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6572800, 12.6602669
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0842590, 16.0846252
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0847397, 15.0889435
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1503830, 13.1556282
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1879463, 8.1886139
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2185707, 11.2200851
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5165119, 8.5171680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1464

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 561

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0938338, upper bound: 5.0827514
time: 102.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0849326, upper bound: 5.0916426
time: 20.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4223862, 13.4244843
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3922234, 6.3935280
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5668831, 8.5669308
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1548538, 11.1554451
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2541695, 9.2543869
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1725235, 12.1734314
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1571808, 11.1559563
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4853668, 12.4870453
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8247833, 9.8265114
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9250336, 12.9259834
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2169151, 13.2179337
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4861450, 7.4866772
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4974976, 18.4972839
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8687668, 12.8688927
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4845657, 21.4878540
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8885880, 8.8892097
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1095657, 10.1101723
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7114563, 21.7135391
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6777611, 10.6771278
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5616970, 7.5609341
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2526054, 10.2522354
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9093361, 8.9091835
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6763268, 8.6758804
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7099648, 7.7100620
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1364708, 8.1362057
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3551788, 9.3558769
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5618858, 14.5608902
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1013527, 9.1010056
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6109848, 9.6102829
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6013641, 8.6015854
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6515236, 7.6526833
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1457520, 9.1455021
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6876984, 10.6859207
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5440483, 14.5415001
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6600838, 10.6572495
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.7011604, 10.6979561
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6866531, 12.6836700
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0844116, 16.0828171
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1186066, 15.1159210
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1901932, 13.1873741
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1939392, 8.1930161
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2374210, 11.2349930
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5126781, 8.5127449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0888798, upper bound: 5.0942141
time: 21.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0890755, upper bound: 5.0939991
time: 19.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4234695, 13.4233932
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3929405, 6.3928127
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5668983, 8.5669155
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1551666, 11.1551361
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2541771, 9.2543793
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1735611, 12.1723938
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1567612, 11.1563759
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4863586, 12.4860535
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8263702, 9.8249207
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9251938, 12.9258194
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2178383, 13.2170067
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4866180, 7.4862041
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4970932, 18.4976959
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8677979, 12.8698692
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4866180, 21.4858093
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8892746, 8.8885212
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1095886, 10.1101494
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7126160, 21.7123718
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6777573, 10.6771317
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5612125, 7.5614223
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2525864, 10.2522545
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9094009, 8.9091187
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6762733, 8.6759338
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7099800, 7.7100468
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1364708, 8.1362057
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3557396, 9.3553200
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5618858, 14.5608902
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1014671, 9.1008911
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6105957, 9.6106720
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6019325, 8.6010208
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6526299, 7.6515713
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1456680, 9.1455860
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6863365, 10.6872864
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5417671, 14.5437813
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6583176, 10.6590157
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6985893, 10.7005272
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6843491, 12.6859703
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0833664, 16.0838623
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1165848, 15.1179466
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1872940, 13.1902733
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1935272, 8.1934261
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2359409, 11.2364769
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5130482, 8.5123749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0913768, upper bound: 5.0868651
time: 19.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0864133, upper bound: 5.0918590
time: 19.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3307152, 13.3278694
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3383923, 6.3354874
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5373001, 8.5375729
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1248894, 11.1254387
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2257957, 9.2257195
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1259766, 12.1246567
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1010170, 11.1019173
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4288101, 12.4258270
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7192192, 9.7152672
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8811798, 12.8809700
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1579742, 13.1572571
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4794445, 7.4771652
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4505157, 18.4519577
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8568497, 12.8596191
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.3741989, 21.3670731
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8525963, 8.8521938
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0711708, 10.0701656
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6372147, 21.6333237
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6649323, 10.6637154
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5529900, 7.5530968
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2402916, 10.2397614
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8917389, 8.8904514
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6499157, 8.6505318
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7102623, 7.7102299
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1360226, 8.1351604
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3599892, 9.3577728
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5482864, 14.5494995
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0893555, 9.0883045
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6121712, 9.6123428
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6019592, 8.6014748
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6639919, 7.6607590
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1293564, 9.1279716
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6011009, 10.6057625
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4361916, 14.4419098
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.5677643, 10.5738564
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.5741577, 10.5811806
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.5522041, 12.5582809
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0273666, 16.0306473
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -14.9918518, 14.9959373
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.0677223, 13.0740013
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1717186, 8.1756420
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1592789, 11.1637573
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5204887, 8.5205307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1368

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1415

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0860722, upper bound: 5.0799388
time: 19.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0860722, upper bound: 5.0799388
time: 18.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3709488, 13.3603249
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3676605, 6.3617592
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5437317, 8.5388565
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1375160, 11.1331978
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2318344, 9.2253990
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1491470, 12.1443253
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1212234, 11.1263199
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4593468, 12.4522820
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7594490, 9.7447701
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8932571, 12.8891830
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1785889, 13.1732788
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4822912, 7.4829102
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4617310, 18.4693756
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8700943, 12.8764915
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4229584, 21.4132843
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8577118, 8.8521900
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0839119, 10.0804405
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6669998, 21.6603088
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6641312, 10.6654549
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5494576, 7.5506363
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2363205, 10.2374306
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8903618, 8.8919144
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6602230, 8.6621342
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7101231, 7.7105885
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1343288, 8.1349125
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3537655, 9.3545990
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5589142, 14.5607910
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0866032, 9.0853081
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6071167, 9.6089630
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6007957, 8.6009369
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6543674, 7.6547222
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1256065, 9.1270027
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6373482, 10.6435928
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4933319, 14.4998322
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6170311, 10.6229019
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6340485, 10.6438103
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6104851, 12.6211777
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0651474, 16.0682907
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0397263, 15.0487518
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1322327, 13.1395149
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1964149, 8.1975136
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1986256, 11.2040100
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5207729, 8.5206089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 689

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0826050, upper bound: 5.0936473
time: 24.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0854831, upper bound: 5.0908430
time: 25.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3690796, 13.3621941
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3664169, 6.3630028
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5417442, 8.5408401
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1368828, 11.1338310
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2288055, 9.2284279
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1480255, 12.1454468
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1212883, 11.1262550
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4568901, 12.4547386
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7550888, 9.7491302
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8944168, 12.8880310
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1791153, 13.1727524
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4825544, 7.4826469
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4649811, 18.4661255
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8750076, 12.8715744
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4229355, 21.4133148
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8567657, 8.8531322
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0846443, 10.0797119
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6674576, 21.6598511
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6629868, 10.6665955
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5488091, 7.5512867
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2351341, 10.2386169
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.8896866, 8.8925896
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6601620, 8.6621914
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7099705, 7.7107449
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1342220, 8.1350231
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3553982, 9.3529701
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5589294, 14.5607758
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0828953, 9.0890141
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6071358, 9.6089401
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6009293, 8.6008072
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6555424, 7.6535492
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1244659, 9.1281433
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6369553, 10.6439819
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4935074, 14.4996605
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6169205, 10.6230164
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6354065, 10.6424484
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6113625, 12.6202965
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0651932, 16.0682526
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0388718, 15.0496101
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1325607, 13.1391830
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1959229, 8.1980057
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1983624, 11.2042732
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5207615, 8.5206223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1385

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 528

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0889734, upper bound: 5.0896978
time: 22.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0881133, upper bound: 5.0905584
time: 20.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4046021, 13.4040108
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3834953, 6.3827877
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5553894, 8.5540771
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1518707, 11.1501999
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2453461, 9.2424469
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1649323, 12.1640244
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1471481, 11.1477432
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4760590, 12.4733849
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7942657, 9.7925491
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9142456, 12.9162216
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2041321, 13.2061768
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4874649, 7.4876671
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4832840, 18.4855576
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8656158, 12.8684044
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4424744, 21.4450378
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8717117, 8.8732491
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1010361, 10.1030350
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6964951, 21.6977234
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6754761, 10.6752129
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5607204, 7.5603714
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2465935, 10.2451363
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9041672, 8.9029121
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6736202, 8.6734772
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7105999, 7.7103367
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1444664, 8.1445408
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3530655, 9.3549690
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5631180, 14.5646515
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1036911, 9.0998878
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6112518, 9.6117859
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6021118, 8.6033096
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6561317, 7.6580467
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1393280, 9.1372948
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6676178, 10.6646461
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5239868, 14.5225487
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6409264, 10.6394310
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6703300, 10.6692505
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6581421, 12.6564980
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0659409, 16.0680695
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0921707, 15.0880928
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1684418, 13.1663933
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1892776, 8.1897144
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2152672, 11.2154121
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5142593, 8.5148487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 956

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 743

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0903669, upper bound: 5.0856377
time: 22.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0850020, upper bound: 5.0909451
time: 18.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4045944, 13.4040146
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3834381, 6.3828430
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5551987, 8.5542679
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1513824, 11.1506920
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2451096, 9.2426834
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1645279, 12.1644211
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1476898, 11.1472015
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4757233, 12.4737167
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7938423, 9.7929726
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9143906, 12.9160805
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2039719, 13.2063370
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4874611, 7.4876690
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4834900, 18.4853516
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8658752, 12.8681412
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4413910, 21.4461212
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8719864, 8.8729725
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1013412, 10.1027260
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6956100, 21.6985931
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6757545, 10.6749344
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5607738, 7.5603142
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2466164, 10.2451134
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9041634, 8.9029160
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6737576, 8.6733379
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7105961, 7.7103405
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1444511, 8.1445599
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3532600, 9.3547745
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5635071, 14.5642624
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1036530, 9.0999241
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6114006, 9.6116371
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.6022949, 8.6031265
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6561470, 7.6580315
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1389847, 9.1376400
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6676140, 10.6646500
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5245972, 14.5219421
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6413116, 10.6390495
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6707840, 10.6687965
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6584320, 12.6562004
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0675583, 16.0664520
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0921783, 15.0880890
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1687469, 13.1660919
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1903343, 8.1886578
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2159843, 11.2146950
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5145950, 8.5145111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1385

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0896907, upper bound: 5.0886800
time: 15.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0867607, upper bound: 5.0915959
time: 17.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4024048, 13.4063187
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3767719, 6.3792076
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5405540, 8.5459957
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1499443, 11.1513367
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2291374, 9.2347183
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1661072, 12.1684875
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1357231, 11.1367455
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4706726, 12.4749222
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7963791, 9.8038101
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9182129, 12.9175148
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1928864, 13.1924286
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4883919, 7.4872322
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4821701, 18.4761353
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8701553, 12.8631287
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4252472, 21.4246750
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8406372, 8.8403816
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0944252, 10.0962410
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6848679, 21.6838684
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6671257, 10.6690483
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5502052, 7.5523453
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2463303, 10.2471695
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9019966, 8.9017010
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6647949, 8.6627045
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7094669, 7.7095165
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1441841, 8.1436634
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3358307, 9.3306046
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5745964, 14.5738754
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0979977, 9.1015682
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6102371, 9.6105156
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5871716, 8.5842934
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6299820, 7.6256027
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1376572, 9.1380501
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6492310, 10.6501083
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5176811, 14.5172501
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6354256, 10.6355858
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6721153, 10.6708145
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6416512, 12.6410370
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0834808, 16.0840149
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0435486, 15.0473747
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1426964, 13.1433105
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1993370, 8.2000275
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1937904, 11.1981125
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5197277, 8.5196953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 545

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0939264, upper bound: 5.0804030
time: 23.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0869500, upper bound: 5.0864134
time: 20.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4031677, 13.4055557
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3773708, 6.3786068
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5431442, 8.5434017
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1512489, 11.1500320
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2313232, 9.2325325
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1667557, 12.1678314
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1368256, 11.1356430
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4723511, 12.4732437
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7963829, 9.8038063
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9179916, 12.9177399
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1912689, 13.1940422
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4875526, 7.4880733
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4805603, 18.4777374
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8704300, 12.8628464
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4209824, 21.4289474
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8372345, 8.8437824
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0960159, 10.0946503
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6826706, 21.6860580
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6673050, 10.6688728
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5514488, 7.5510998
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2462769, 10.2472267
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9015999, 8.9020977
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6628723, 8.6646271
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7094631, 7.7095203
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1433525, 8.1444988
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3321571, 9.3342781
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5727730, 14.5757065
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0979252, 9.1016388
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6101761, 9.6105766
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5835438, 8.5879211
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6260490, 7.6295338
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1374588, 9.1382484
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6522980, 10.6470413
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5196457, 14.5152893
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6372375, 10.6337738
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6750679, 10.6678619
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6447411, 12.6379509
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0835953, 16.0839081
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0497437, 15.0411797
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1463280, 13.1396828
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1996613, 8.1997032
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1981621, 11.1937408
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5194149, 8.5200043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 641

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0884441, upper bound: 5.0852745
time: 11.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0884441, upper bound: 5.0852745
time: 12.04 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0931192, upper bound: 5.0839136
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0906360, upper bound: 5.0863601
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0937633, upper bound: 5.0858413
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0937548, upper bound: 5.0858497
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0844533, upper bound: 5.0909112
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0856447, upper bound: 5.0897266
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0815611, upper bound: 5.0930052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0815611, upper bound: 5.0911227
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0935912, upper bound: 5.0861309
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0883924, upper bound: 5.0913737
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0938338, upper bound: 5.0827514
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0849326, upper bound: 5.0916426
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0888798, upper bound: 5.0942141
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0890755, upper bound: 5.0939991
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0913768, upper bound: 5.0868651
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0864133, upper bound: 5.0918590
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0860722, upper bound: 5.0799388
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0860722, upper bound: 5.0799388
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0826050, upper bound: 5.0936473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0854831, upper bound: 5.0908430
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0889734, upper bound: 5.0896978
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0881133, upper bound: 5.0905584
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0903669, upper bound: 5.0856377
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0850020, upper bound: 5.0909451
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0896907, upper bound: 5.0886800
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0867607, upper bound: 5.0915959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0939264, upper bound: 5.0804030
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0869500, upper bound: 5.0864134
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0884441, upper bound: 5.0852745
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 11, lower bound: -5.0884441, upper bound: 5.0852745

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4102592, 13.4143715
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3838177, 6.3865967
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5621262, 8.5636120
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1475067, 11.1505241
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2463837, 9.2494240
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1689758, 12.1713715
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1425400, 11.1404419
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4765472, 12.4797821
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8081207, 9.8127136
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9114838, 12.9139671
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2106857, 13.2119102
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4830418, 7.4815331
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4922256, 18.4901199
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8538132, 12.8545570
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4776917, 21.4793472
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8876953, 8.8880959
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1011314, 10.1035099
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7031708, 21.7048340
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6694031, 10.6676216
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5596313, 7.5586319
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2471352, 10.2463913
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9074821, 8.9077988
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6712589, 8.6704140
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7093506, 7.7079620
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1357975, 8.1345406
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3459034, 9.3454094
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5544930, 14.5518188
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0936604, 9.0924416
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6132774, 9.6116409
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5999432, 8.5980377
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6498222, 7.6486359
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1435280, 9.1430225
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6821747, 10.6814861
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5410614, 14.5410767
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6514244, 10.6504135
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6944389, 10.6932068
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6780586, 12.6757126
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0855255, 16.0840454
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1095734, 15.1087265
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1891098, 13.1889458
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1867294, 8.1858768
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2245636, 11.2221527
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5126534, 8.5127087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0927779, upper bound: 5.0740859
time: 15.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0832914, upper bound: 5.0835722
time: 20.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4104881, 13.4141388
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3840542, 6.3863564
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5621300, 8.5636082
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1478577, 11.1501732
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2464867, 9.2493210
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1691895, 12.1711655
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1426353, 11.1403465
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4766159, 12.4797134
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8087959, 9.8120422
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9113312, 12.9141121
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2107925, 13.2118034
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4830379, 7.4815369
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4917831, 18.4905624
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8534088, 12.8549576
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4773865, 21.4796600
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8878555, 8.8879337
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1011391, 10.1035061
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7029419, 21.7050629
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6693954, 10.6676292
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5595665, 7.5586948
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2471237, 10.2464027
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9074631, 8.9078159
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6712513, 8.6704178
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7093315, 7.7079811
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1359653, 8.1343746
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3461170, 9.3451958
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5544395, 14.5518646
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0938320, 9.0922680
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6132736, 9.6116409
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5998783, 8.5980988
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6497536, 7.6487045
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1435661, 9.1429863
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6822586, 10.6814041
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5413284, 14.5408096
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6519852, 10.6498489
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6945915, 10.6930580
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6780586, 12.6757164
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0857086, 16.0838547
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1099091, 15.1083908
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1892090, 13.1888466
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1870728, 8.1855316
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2246208, 11.2220993
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5132370, 8.5121250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 663

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 529

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0906098, upper bound: 5.0841721
time: 22.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0884331, upper bound: 5.0863340
time: 19.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3878708, 13.3930206
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3763962, 6.3792610
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5591965, 8.5608959
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1500435, 11.1510201
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2473183, 9.2501144
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1675110, 12.1695442
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1125031, 11.1098480
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4752426, 12.4774094
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7863808, 9.7926598
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8945618, 12.8977089
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1883469, 13.1916885
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4819622, 7.4804745
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4819946, 18.4790115
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8498688, 12.8458290
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4593811, 21.4622650
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8616791, 8.8654900
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.0999565, 10.1019936
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6996613, 21.7002182
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6674728, 10.6672897
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5624084, 7.5621128
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2442322, 10.2439995
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9032745, 8.9030495
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6703224, 8.6699276
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7025890, 7.7020168
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1377525, 8.1370792
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3409729, 9.3398628
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5461197, 14.5470886
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0933285, 9.0937290
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6110420, 9.6110153
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5909996, 8.5914192
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6502075, 7.6494713
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1386719, 9.1364689
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6613541, 10.6582603
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5162621, 14.5129700
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6395836, 10.6385345
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6699944, 10.6670380
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6518402, 12.6470108
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0845871, 16.0842743
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0776901, 15.0737190
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1549835, 13.1491280
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1796036, 8.1794853
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2062874, 11.2048073
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5150528, 8.5147152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1465

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 512

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0937526, upper bound: 5.0858321
time: 18.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0937541, upper bound: 5.0858306
time: 19.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3908463, 13.3900452
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3776169, 6.3780403
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5594864, 8.5606098
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1501427, 11.1509247
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2474060, 9.2500267
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1677704, 12.1692696
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1085663, 11.1137886
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4755859, 12.4770622
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7898788, 9.7891579
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8966370, 12.8956299
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1913071, 13.1887283
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4820004, 7.4804363
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4800568, 18.4809494
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8477097, 12.8479805
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4619904, 21.4596558
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8653030, 8.8618698
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1001244, 10.1018257
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7005157, 21.6993561
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6667290, 10.6680336
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5616608, 7.5628586
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2438278, 10.2444038
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9023590, 8.9039650
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6703873, 8.6698627
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7032223, 7.7013836
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1377602, 8.1370716
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3418961, 9.3389397
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5469513, 14.5462494
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0932293, 9.0938282
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6110764, 9.6109810
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5920715, 8.5903473
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6505585, 7.6491241
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1369705, 9.1381683
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6582375, 10.6613808
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5127106, 14.5165253
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6378784, 10.6402359
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6666260, 10.6704063
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6481628, 12.6506882
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0840530, 16.0848083
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0734940, 15.0779114
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1500168, 13.1540947
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1791115, 8.1799774
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2039375, 11.2071571
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5152626, 8.5145073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 743

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1776

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0910610, upper bound: 5.0858497
time: 19.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0937548, upper bound: 5.0831522
time: 43.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4257889, 13.4229774
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3939896, 6.3930779
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5629692, 8.5615616
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1536217, 11.1528358
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2431984, 9.2423134
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1697998, 12.1680527
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1494598, 11.1481171
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4826050, 12.4818535
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8252487, 9.8231468
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9207611, 12.9192009
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2151337, 13.2137642
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4808140, 7.4824409
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4959793, 18.4958572
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8664322, 12.8678703
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4891052, 21.4889145
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8898315, 8.8902073
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1085434, 10.1065235
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7144928, 21.7146530
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6743698, 10.6748543
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5547142, 7.5557117
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2509575, 10.2514610
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9050446, 8.9058456
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6707554, 8.6722393
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7056522, 7.7066269
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1281738, 8.1306019
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3485336, 9.3517036
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5587540, 14.5583344
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0977669, 9.0989361
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6071663, 9.6075134
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5987625, 8.5992279
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6486778, 7.6501484
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1353149, 9.1377773
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6852875, 10.6844978
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5425797, 14.5432396
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6587791, 10.6582298
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6982574, 10.6986923
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6845093, 12.6854706
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0798035, 16.0792084
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1152954, 15.1155090
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1847992, 13.1873589
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1878357, 8.1860313
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2334328, 11.2323685
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5019093, 8.5001640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1465

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0840171, upper bound: 5.0851989
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0789307, upper bound: 5.0904615
time: 22.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4229507, 13.4219437
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3921242, 6.3910484
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5622749, 8.5606861
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1527557, 11.1528511
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2428627, 9.2399139
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1691132, 12.1680222
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1531525, 11.1521797
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4804230, 12.4793015
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8222313, 9.8193398
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9202652, 12.9198799
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2149963, 13.2152252
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4809179, 7.4824429
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4945831, 18.4965744
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8648682, 12.8682671
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4882812, 21.4888306
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8905754, 8.8894997
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1078529, 10.1066551
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7134705, 21.7146683
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6758041, 10.6743698
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5572681, 7.5559521
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2514229, 10.2509842
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9078865, 8.9072647
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6720428, 8.6721363
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7056999, 7.7064476
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1301346, 8.1297359
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3496628, 9.3509979
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5577087, 14.5560837
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0991211, 9.0973549
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6082840, 9.6078110
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5985432, 8.5988426
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6488953, 7.6498089
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1402626, 9.1396027
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6848640, 10.6849346
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5429993, 14.5422134
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6599998, 10.6590996
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6961327, 10.6957283
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6827698, 12.6836891
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0788803, 16.0773010
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1145630, 15.1146164
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1855087, 13.1851845
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1918526, 8.1904240
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2353859, 11.2332077
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5031242, 8.5038395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1465

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0811160, upper bound: 5.0873885
time: 10.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0758895, upper bound: 5.0925656
time: 21.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4227524, 13.4221382
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3919487, 6.3912239
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5621872, 8.5607738
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1527252, 11.1528816
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2426758, 9.2401009
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1690521, 12.1680832
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1531792, 11.1521530
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4801407, 12.4795837
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.8219223, 9.8196487
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9202881, 12.9198608
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.2150955, 13.2151299
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4809675, 7.4823952
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4947739, 18.4963837
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8652496, 12.8678818
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4882660, 21.4888458
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8905563, 8.8895168
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1078568, 10.1066513
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.7134705, 21.7146683
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6757927, 10.6743851
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5572186, 7.5560017
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2513046, 10.2511024
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9078636, 8.9072876
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6720352, 8.6721420
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.7058067, 7.7063389
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1301727, 8.1296997
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3497581, 9.3509026
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5578461, 14.5559387
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0988731, 9.0976048
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6083145, 9.6077805
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5986004, 8.5987892
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6489105, 7.6497955
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1402435, 9.1396236
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6847305, 10.6850719
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.5430679, 14.5421448
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6600037, 10.6590958
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6963882, 10.6954727
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6829758, 12.6834793
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0790405, 16.0771408
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.1144714, 15.1147041
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1856918, 13.1849976
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1918526, 8.1904259
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2353745, 11.2332191
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5031357, 8.5038242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 754

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0825487, upper bound: 5.0905214
time: 9.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0828499, upper bound: 5.0902210
time: 23.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3799667, 13.3818626
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3693218, 6.3704758
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5448647, 8.5433655
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1368942, 11.1367188
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2459679, 9.2453270
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1486893, 12.1472740
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1288376, 11.1243248
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4552078, 12.4541740
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7719078, 9.7751427
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8987656, 12.9017677
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1921082, 13.1953125
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4764423, 7.4760246
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4603043, 18.4592743
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8614426, 12.8599548
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4613724, 21.4640961
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8422012, 8.8484612
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1089935, 10.1095371
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6966324, 21.6965942
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6723328, 10.6712227
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5642567, 7.5630970
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2504044, 10.2497406
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9064751, 8.9052525
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6602249, 8.6610432
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6994400, 7.7004242
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1355095, 8.1355419
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3412933, 9.3443222
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5397110, 14.5408936
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0965900, 9.0964775
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6104126, 9.6107635
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5842228, 8.5861740
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6503334, 7.6517239
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1362152, 9.1342545
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6646805, 10.6613426
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4942474, 14.4935646
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6400566, 10.6390915
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6528893, 10.6512337
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6484604, 12.6452408
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0528107, 16.0542221
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0816574, 15.0766335
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1405449, 13.1353989
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1768074, 8.1774940
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2208576, 11.2176781
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5122452, 8.5135975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1675

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1464

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0934521, upper bound: 5.0852245
time: 17.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0924442, upper bound: 5.0859948
time: 20.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.3774109, 13.3844185
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3681850, 6.3716125
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5430336, 8.5451965
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1356430, 11.1379738
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2448273, 9.2464638
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1467361, 12.1492348
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.1288834, 11.1242790
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4530106, 12.4563751
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7701530, 9.7768936
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.8991928, 12.9013329
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1920395, 13.1953773
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4759312, 7.4765377
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4619064, 18.4576721
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8618088, 12.8595886
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4608688, 21.4645996
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8445778, 8.8460808
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1087761, 10.1097584
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6952896, 21.6979523
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6726112, 10.6709442
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5642414, 7.5631104
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2505150, 10.2496338
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9064178, 8.9053078
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6613846, 8.6598854
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6997452, 7.7001171
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1354675, 8.1355858
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3428345, 9.3427811
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5411606, 14.5394440
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.0966969, 9.0963707
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6110420, 9.6101341
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5855694, 8.5848236
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6514626, 7.6505947
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1360474, 9.1344261
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6647072, 10.6613159
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4968529, 14.4909630
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6407433, 10.6384048
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6549339, 10.6491928
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.6496048, 12.6440964
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0549316, 16.0521011
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0808182, 15.0774689
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.1400871, 13.1358566
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1778069, 8.1764908
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.2208652, 11.2176743
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5125237, 8.5133209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0876412, upper bound: 5.0911570
time: 19.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0881711, upper bound: 5.0904762
time: 20.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.2282829, 0.7062590, -20.2282829, 0.7062590, -13.4007530, 13.3990860
1: -6.4244299, 5.2899771, -6.4244299, 5.2899771, -6.3789139, 6.3780403
2: -11.0250292, 2.2252476, -11.0250292, 2.2252476, -8.5522461, 8.5542660
3: -12.3506584, 3.3957181, -12.3506584, 3.3957181, -11.1477203, 11.1495171
4: -22.0544357, -5.6413059, -22.0544357, -5.6413059, -9.2510147, 9.2536812
5: -10.8085623, 5.6431475, -10.8085623, 5.6431475, -12.1679535, 12.1676559
6: -22.4935303, -4.5087814, -22.4935303, -4.5087814, -11.0970764, 11.1038322
7: -9.5688019, 8.9286137, -9.5688019, 8.9286137, -12.4844360, 12.4844475
8: -26.3516388, -5.5972786, -26.3516388, -5.5972786, -9.7814217, 9.7766685
9: -14.5972672, 2.1422729, -14.5972672, 2.1422729, -12.9012909, 12.8994980
10: -5.9042730, 11.7204475, -5.9042730, 11.7204475, -13.1712036, 13.1652298
11: 9.5967846, 21.1215839, 9.5967846, 21.1215839, -7.4657383, 7.4629936
12: -15.1702166, 9.8695765, -15.1702166, 9.8695765, -18.4792938, 18.4810715
13: -28.0377922, -3.0761464, -28.0377922, -3.0761464, -12.8383408, 12.8454170
14: -31.3603611, 0.6722455, -31.3603611, 0.6722455, -21.4264374, 21.4187164
15: -24.9375725, -10.6921339, -24.9375725, -10.6921339, -8.8365250, 8.8300114
16: -6.9504609, 7.8883667, -6.9504609, 7.8883667, -10.1060524, 10.1062546
17: -14.7425632, 11.7812157, -14.7425632, 11.7812157, -21.6987762, 21.6957016
18: -0.8761041, 12.5853157, -0.8761041, 12.5853157, -10.6755257, 10.6754646
19: -5.2752228, 4.7464051, -5.2752228, 4.7464051, -7.5585270, 7.5592499
20: -3.4029522, 7.9701266, -3.4029522, 7.9701266, -10.2488289, 10.2488518
21: -1.9313600, 8.8826933, -1.9313600, 8.8826933, -8.9018936, 8.9019928
22: -9.2060995, 2.7860768, -9.2060995, 2.7860768, -8.6757126, 8.6749458
23: 1.3802859, 12.5147171, 1.3802859, 12.5147171, -7.6943684, 7.6919861
24: -2.6498592, 10.5029688, -2.6498592, 10.5029688, -8.1223984, 8.1204681
25: 0.3958333, 13.7677917, 0.3958333, 13.7677917, -9.3105545, 9.3048172
26: -17.3860321, 2.4882891, -17.3860321, 2.4882891, -14.5463905, 14.5429764
27: -10.2686863, 6.2961550, -10.2686863, 6.2961550, -9.1063213, 9.1052322
28: 1.1113470, 13.5833578, 1.1113470, 13.5833578, -9.6100807, 9.6086998
29: -5.0845928, 8.3623238, -5.0845928, 8.3623238, -8.5866795, 8.5820198
30: 5.9989243, 17.7221909, 5.9989243, 17.7221909, -7.6152782, 7.6095676
31: -3.3616734, 10.4007454, -3.3616734, 10.4007454, -9.1375694, 9.1389580
32: -19.5967464, -2.7746017, -19.5967464, -2.7746017, -10.6133728, 10.6222954
33: -47.0481148, -21.5724850, -47.0481148, -21.5724850, -14.4635086, 14.4738464
34: -29.7305546, -10.5938530, -29.7305546, -10.5938530, -10.6078224, 10.6141586
35: -29.2326946, -9.9649715, -29.2326946, -9.9649715, -10.6173668, 10.6273346
36: -31.8955345, -9.4200172, -31.8955345, -9.4200172, -12.5963211, 12.6068115
37: -46.1406631, -23.4843807, -46.1406631, -23.4843807, -16.0729599, 16.0752945
38: -34.3056984, -11.5219917, -34.3056984, -11.5219917, -15.0089569, 15.0224953
39: -56.3328514, -30.6840725, -56.3328514, -30.6840725, -13.0698471, 13.0847664
40: -40.2598648, -23.3088779, -40.2598648, -23.3088779, -8.1766930, 8.1787548
41: -26.7461548, -7.0015345, -26.7461548, -7.0015345, -11.1735077, 11.1805725
42: -14.5394669, -2.1234100, -14.5394669, -2.1234100, -8.5257568, 8.5254707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=90, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1351

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 839

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0829147, upper bound: 5.0825482
time: 18.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0936292, upper bound: 5.0718892
time: 23.27 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 44.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0927779, upper bound: 5.0740859
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0832914, upper bound: 5.0835722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0906098, upper bound: 5.0841721
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0884331, upper bound: 5.0863340
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0937526, upper bound: 5.0858321
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0937541, upper bound: 5.0858306
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0910610, upper bound: 5.0858497
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0937548, upper bound: 5.0831522
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0840171, upper bound: 5.0851989
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0789307, upper bound: 5.0904615
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0811160, upper bound: 5.0873885
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0758895, upper bound: 5.0925656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0825487, upper bound: 5.0905214
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0828499, upper bound: 5.0902210
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0934521, upper bound: 5.0852245
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0924442, upper bound: 5.0859948
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0876412, upper bound: 5.0911570
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0881711, upper bound: 5.0904762
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0829147, upper bound: 5.0825482
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 44.31
Output dim: 11, lower bound: -5.0936292, upper bound: 5.0718892
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0849326, upper bound: 5.0916426
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0888798, upper bound: 5.0942141
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0890755, upper bound: 5.0939991
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0913768, upper bound: 5.0868651
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0864133, upper bound: 5.0918590
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0826050, upper bound: 5.0936473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0854831, upper bound: 5.0908430
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0881133, upper bound: 5.0905584
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0903669, upper bound: 5.0856377
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0850020, upper bound: 5.0909451
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0867607, upper bound: 5.0915959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.31
Output dim: 11, lower bound: -5.0939264, upper bound: 5.0804030

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 29.04 + 1811.64 = 1840.68 seconds
