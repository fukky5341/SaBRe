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
execution time: IAR + RelationalAnalysis = 2.81 + 26.70 = 29.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -5.0948664, upper bound: 5.0948664

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 625

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0945278, upper bound: 5.0846100
time: 18.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0945278, upper bound: 5.0946300
time: 21.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 40.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 40.11
Output dim: 11, lower bound: -5.0945278, upper bound: 5.0846100
IS_A2, status: Status.UNKNOWN, split count: 1, time: 40.11
Output dim: 11, lower bound: -5.0945278, upper bound: 5.0946300

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -20.2264595, 0.7002201, -20.2272892, 0.7030122, -13.4280319, 13.4236107
1: -6.4237776, 5.2847757, -6.4240880, 5.2871718, -6.3970699, 6.3937855
2: -11.0153942, 2.2244608, -11.0198221, 2.2248251, -8.5567856, 8.5607052
3: -12.3443766, 3.3948541, -12.3472538, 3.3952432, -11.1551437, 11.1575356
4: -22.0458565, -5.6426563, -22.0498295, -5.6420426, -9.2500801, 9.2532997
5: -10.7967653, 5.6419010, -10.8021984, 5.6424608, -12.1630096, 12.1679230
6: -22.4667530, -4.5092831, -22.4790840, -4.5090313, -11.1347198, 11.1469345
7: -9.5646534, 8.9258928, -9.5665436, 8.9271688, -12.4876862, 12.4876213
8: -26.3478241, -5.5982552, -26.3495846, -5.5978141, -9.8318253, 9.8316650
9: -14.5959196, 2.1394577, -14.5965271, 2.1407275, -12.9264908, 12.9255486
10: -5.9023657, 11.7180214, -5.9032421, 11.7191753, -13.2174835, 13.2169800
11: 9.5997581, 21.1037903, 9.5984001, 21.1119652, -7.4744167, 7.4681358
12: -15.1623383, 9.8679447, -15.1659355, 9.8686724, -18.4938889, 18.4972687
13: -28.0358143, -3.0845101, -28.0366955, -3.0806720, -12.8748283, 12.8717117
14: -31.3572693, 0.6610503, -31.3587074, 0.6662476, -21.4852448, 21.4796677
15: -24.9366302, -10.6957130, -24.9370747, -10.6941051, -8.8888245, 8.8873520
16: -6.9485321, 7.8855252, -6.9493861, 7.8868098, -10.1104813, 10.1088409
17: -14.7404013, 11.7739258, -14.7413979, 11.7771950, -21.7135010, 21.7095032
18: -0.8742661, 12.5827570, -0.8751068, 12.5838871, -10.6792107, 10.6791534
19: -5.2735071, 4.7428432, -5.2743258, 4.7444792, -7.5581417, 7.5567093
20: -3.3976181, 7.9692678, -3.4000645, 7.9696555, -10.2480927, 10.2508888
21: -1.9288468, 8.8793592, -1.9300146, 8.8808584, -8.9055824, 8.9053535
22: -9.2044563, 2.7831628, -9.2052221, 2.7844810, -8.6744881, 8.6738262
23: 1.3820251, 12.5018263, 1.3812156, 12.5077686, -7.7026596, 7.6973305
24: -2.6486750, 10.4853716, -2.6492231, 10.4933758, -8.1258812, 8.1175117
25: 0.3974264, 13.7390623, 0.3966885, 13.7522993, -9.3436375, 9.3306923
26: -17.3832779, 2.4872599, -17.3846130, 2.4877362, -14.5664520, 14.5676956
27: -10.2655048, 6.2942543, -10.2669735, 6.2951345, -9.1020813, 9.1037216
28: 1.1133957, 13.5739222, 1.1124561, 13.5782356, -9.6024132, 9.5986290
29: -5.0832653, 8.3589983, -5.0838785, 8.3605480, -8.6018410, 8.6007805
30: 6.0011363, 17.7055359, 6.0001211, 17.7131157, -7.6419373, 7.6357651
31: -3.3592553, 10.3922157, -3.3603656, 10.3961239, -9.1402893, 9.1373024
32: -19.5727711, -2.7755420, -19.5833340, -2.7751045, -10.6655540, 10.6772976
33: -47.0436783, -21.5749626, -47.0457611, -21.5738525, -14.5391884, 14.5396538
34: -29.7231674, -10.5950298, -29.7265129, -10.5944901, -10.6554146, 10.6590691
35: -29.2284164, -9.9666920, -29.2303963, -9.9658670, -10.6966019, 10.6981354
36: -31.8875656, -9.4216347, -31.8912296, -9.4208794, -12.6814537, 12.6853828
37: -46.1370926, -23.4900112, -46.1387177, -23.4875679, -16.0733719, 16.0695953
38: -34.2992706, -11.5232220, -34.3020630, -11.5226860, -15.1114349, 15.1140137
39: -56.3292694, -30.6874352, -56.3309402, -30.6859245, -13.1866379, 13.1870995
40: -40.2477531, -23.3091068, -40.2533531, -23.3089828, -8.1857452, 8.1910934
41: -26.7272644, -7.0026464, -26.7359581, -7.0021362, -11.2255058, 11.2337379
42: -14.5354309, -2.1242800, -14.5372658, -2.1238813, -8.5091248, 8.5105381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1465

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0941126, upper bound: 5.0790879
time: 20.84 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0941126, upper bound: 5.0842004
time: 14.00 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -20.2349720, 0.7055006, -20.2277431, 0.7009876, -13.4555550, 13.4385223
1: -6.4308186, 5.2898073, -6.4242783, 5.2858906, -6.4174423, 6.4041882
2: -11.0260763, 2.2461367, -11.0242195, 2.2250650, -8.5649033, 8.5875530
3: -12.3532400, 3.4096668, -12.3494749, 3.3954296, -11.1638489, 11.1762390
4: -22.0552826, -5.6236019, -22.0538025, -5.6416607, -9.2566910, 9.2770882
5: -10.8092804, 5.6686783, -10.8076591, 5.6428366, -12.1731720, 12.1996651
6: -22.4931736, -4.4648199, -22.4914932, -4.5089684, -11.1554642, 11.2040138
7: -9.5700550, 8.9340839, -9.5680027, 8.9282589, -12.4941635, 12.4953880
8: -26.3549404, -5.5923295, -26.3509941, -5.5975065, -9.8452950, 9.8345718
9: -14.6024446, 2.1476021, -14.5969429, 2.1410646, -12.9365005, 12.9345779
10: -5.9115272, 11.7293892, -5.9038124, 11.7199326, -13.2290497, 13.2285576
11: 9.5655565, 21.1211891, 9.5971718, 21.1202946, -7.5184584, 7.4819508
12: -15.1725283, 9.8764353, -15.1673870, 9.8691854, -18.5067825, 18.5118637
13: -28.0501518, -3.0720937, -28.0373325, -3.0776660, -12.8936157, 12.8846397
14: -31.3900127, 0.6710746, -31.3596153, 0.6694579, -21.5324326, 21.4957962
15: -24.9417305, -10.6876717, -24.9373417, -10.6933441, -8.8978539, 8.8965664
16: -6.9521160, 7.8919611, -6.9500260, 7.8861418, -10.1254578, 10.1177788
17: -14.7657099, 11.7816477, -14.7421627, 11.7803650, -21.7460175, 21.7211304
18: -0.8860676, 12.5893669, -0.8754976, 12.5845947, -10.6941986, 10.6870308
19: -5.2888336, 4.7460284, -5.2749596, 4.7451954, -7.5772629, 7.5611954
20: -3.4081478, 7.9826584, -3.4000380, 7.9699092, -10.2605629, 10.2694969
21: -1.9450421, 8.8859377, -1.9310603, 8.8818274, -8.9217110, 8.9134312
22: -9.2124367, 2.7869635, -9.2057552, 2.7855804, -8.6842613, 8.6805973
23: 1.3494326, 12.5140285, 1.3807390, 12.5138416, -7.7424202, 7.7065182
24: -2.6872466, 10.5016489, -2.6495972, 10.5008907, -8.1770668, 8.1295662
25: 0.3370862, 13.7662888, 0.3960919, 13.7659264, -9.4190712, 9.3510933
26: -17.3996830, 2.4939010, -17.3855038, 2.4880240, -14.5857086, 14.5750427
27: -10.2692299, 6.3026819, -10.2665071, 6.2958217, -9.1086845, 9.1164322
28: 1.0844665, 13.5832500, 1.1117735, 13.5827484, -9.6368561, 9.6064606
29: -5.0921202, 8.3641701, -5.0843954, 8.3616791, -8.6128616, 8.6064301
30: 5.9692698, 17.7207737, 5.9992690, 17.7200909, -7.6821823, 7.6480408
31: -3.3828607, 10.4013042, -3.3612494, 10.3997250, -9.1676750, 9.1459503
32: -19.5972366, -2.7368686, -19.5943470, -2.7748144, -10.6858406, 10.7328892
33: -47.0527039, -21.5642242, -47.0476189, -21.5731087, -14.5508995, 14.5524063
34: -29.7318764, -10.5871572, -29.7296162, -10.5940514, -10.6632004, 10.6748161
35: -29.2351875, -9.9599800, -29.2320824, -9.9655094, -10.7033730, 10.7111168
36: -31.8910446, -9.4107723, -31.8910847, -9.4202948, -12.6904182, 12.7076759
37: -46.1481781, -23.4876785, -46.1399422, -23.4890099, -16.0982971, 16.0792465
38: -34.3079758, -11.5167160, -34.3039017, -11.5224133, -15.1210251, 15.1250877
39: -56.3373108, -30.6830711, -56.3322716, -30.6857128, -13.1946640, 13.1937447
40: -40.2602692, -23.2959442, -40.2586594, -23.3088913, -8.1952934, 8.2094498
41: -26.7458191, -6.9703908, -26.7446384, -7.0017433, -11.2412434, 11.2749290
42: -14.5439854, -2.1192651, -14.5386715, -2.1236217, -8.5182877, 8.5172367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1465

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0941126, upper bound: 5.0790879
time: 19.98 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0941126, upper bound: 5.0942122
time: 36.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 58.59 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 58.59
Output dim: 11, lower bound: -5.0941126, upper bound: 5.0790879
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 58.59
Output dim: 11, lower bound: -5.0941126, upper bound: 5.0842004
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 58.59
Output dim: 11, lower bound: -5.0941126, upper bound: 5.0790879
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 58.59
Output dim: 11, lower bound: -5.0941126, upper bound: 5.0942122

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -20.2159595, 0.6755857, -20.2265282, 0.6897495, -13.4030914, 13.3985291
1: -6.4183559, 5.2743425, -6.4237232, 5.2811222, -6.3821354, 6.3830280
2: -11.0072250, 2.2059810, -11.0194483, 2.2138593, -8.5344467, 8.5419254
3: -12.3377934, 3.3818889, -12.3466434, 3.3874378, -11.1359100, 11.1425896
4: -22.0425415, -5.6535473, -22.0497684, -5.6481419, -9.2420006, 9.2440567
5: -10.7867813, 5.6196647, -10.8016882, 5.6294641, -12.1371307, 12.1457405
6: -22.4609737, -4.5094295, -22.4770050, -4.5091124, -11.1262245, 11.1445923
7: -9.5488462, 8.9023781, -9.5650167, 8.9131241, -12.4484329, 12.4627190
8: -26.3377285, -5.6129904, -26.3487587, -5.6067290, -9.8028488, 9.8145905
9: -14.5861511, 2.1334064, -14.5907478, 2.1391504, -12.9153442, 12.9118156
10: -5.8960991, 11.7125378, -5.8997784, 11.7174082, -13.2094688, 13.2073479
11: 9.6229229, 21.0969677, 9.6009035, 21.1078835, -7.4486141, 7.4601917
12: -15.1413898, 9.8606348, -15.1539917, 9.8680973, -18.4710770, 18.4694824
13: -28.0318871, -3.0874667, -28.0344696, -3.0825975, -12.8661118, 12.8664169
14: -31.3468666, 0.6408124, -31.3568802, 0.6541953, -21.4633026, 21.4579773
15: -24.9218292, -10.7182198, -24.9282093, -10.6956453, -8.8758774, 8.8614578
16: -6.9434915, 7.8845706, -6.9482818, 7.8865438, -10.1019363, 10.1057529
17: -14.7322807, 11.7556171, -14.7406960, 11.7668629, -21.6999130, 21.6929779
18: -0.8742945, 12.5767822, -0.8740582, 12.5809345, -10.6742058, 10.6713066
19: -5.2697182, 4.7388020, -5.2731695, 4.7426019, -7.5516434, 7.5513153
20: -3.3901665, 7.9653707, -3.3982301, 7.9673862, -10.2359161, 10.2446480
21: -1.9249375, 8.8793316, -1.9285831, 8.8807373, -8.8986130, 8.9031086
22: -9.1915703, 2.7630677, -9.1977596, 2.7834034, -8.6626320, 8.6479473
23: 1.3886360, 12.4958572, 1.3824006, 12.5047722, -7.6937981, 7.6896935
24: -2.6478353, 10.4820070, -2.6485894, 10.4918690, -8.1234894, 8.1133022
25: 0.4046290, 13.7314415, 0.4006212, 13.7515717, -9.3364716, 9.3193283
26: -17.3671989, 2.4803233, -17.3755264, 2.4871414, -14.5507126, 14.5496674
27: -10.2639523, 6.2901254, -10.2657652, 6.2927866, -9.0957642, 9.0968266
28: 1.1179929, 13.5731478, 1.1145024, 13.5780706, -9.5921478, 9.5936928
29: -5.0762253, 8.3492928, -5.0797439, 8.3599043, -8.5946007, 8.5897141
30: 6.0090747, 17.7019653, 6.0034194, 17.7129364, -7.6331787, 7.6322327
31: -3.3544967, 10.3847656, -3.3594224, 10.3923988, -9.1307640, 9.1286011
32: -19.5676193, -2.7761068, -19.5817432, -2.7754688, -10.6598320, 10.6744652
33: -47.0215073, -21.5954704, -47.0326653, -21.5755424, -14.5161934, 14.5042648
34: -29.7105503, -10.6021433, -29.7189999, -10.5950985, -10.6422005, 10.6455994
35: -29.2134552, -9.9798040, -29.2215004, -9.9664459, -10.6793213, 10.6715126
36: -31.8781891, -9.4256763, -31.8855877, -9.4211655, -12.6717033, 12.6740036
37: -46.1139450, -23.5013599, -46.1249847, -23.4886189, -16.0461121, 16.0336990
38: -34.2969551, -11.5337524, -34.3014565, -11.5286827, -15.1032410, 15.1025696
39: -56.3119278, -30.7005463, -56.3200607, -30.6873913, -13.1690674, 13.1644592
40: -40.2330322, -23.3143139, -40.2457504, -23.3092957, -8.1712608, 8.1773262
41: -26.7189064, -7.0053539, -26.7317924, -7.0028429, -11.2163429, 11.2264900
42: -14.5265379, -2.1270728, -14.5333729, -2.1245091, -8.4990292, 8.5024261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0931027, upper bound: 5.0739692
time: 28.51 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0931027, upper bound: 5.0780614
time: 8.45 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -20.2259445, 0.6934063, -20.2269554, 0.6989741, -13.4236374, 13.4022484
1: -6.4236474, 5.2815728, -6.4239979, 5.2852659, -6.3962898, 6.3793621
2: -11.0152111, 2.2185888, -11.0197392, 2.2213609, -8.5535736, 8.5394669
3: -12.3440189, 3.3906052, -12.3470211, 3.3927479, -11.1536026, 11.1396637
4: -22.0456772, -5.6463652, -22.0497322, -5.6442175, -9.2470627, 9.2447624
5: -10.7963638, 5.6353426, -10.8019629, 5.6386013, -12.1592865, 12.1442375
6: -22.4659023, -4.5095654, -22.4786224, -4.5092092, -11.1336174, 11.1455307
7: -9.5641499, 8.9183636, -9.5662565, 8.9227352, -12.4847946, 12.4566650
8: -26.3476067, -5.6037941, -26.3494339, -5.6010914, -9.8310242, 9.8049812
9: -14.5911322, 2.1386633, -14.5937195, 2.1402600, -12.9153290, 12.9223671
10: -5.9002151, 11.7170362, -5.9019003, 11.7185822, -13.2134666, 13.2148743
11: 9.6004028, 21.1017609, 9.5987806, 21.1107845, -7.4731312, 7.4581871
12: -15.1542330, 9.8676109, -15.1611309, 9.8684807, -18.4688110, 18.4941177
13: -28.0316410, -3.0852213, -28.0342350, -3.0810938, -12.8745804, 12.8658218
14: -31.3556976, 0.6546738, -31.3577919, 0.6625042, -21.4802628, 21.4690857
15: -24.9354897, -10.6962490, -24.9363842, -10.6943970, -8.8722496, 8.8858337
16: -6.9481149, 7.8853807, -6.9491372, 7.8867087, -10.1102448, 10.1065941
17: -14.7396202, 11.7687073, -14.7409229, 11.7741785, -21.7091064, 21.6975174
18: -0.8735771, 12.5784779, -0.8746834, 12.5813961, -10.6766968, 10.6734352
19: -5.2729282, 4.7416458, -5.2739463, 4.7437611, -7.5569363, 7.5546474
20: -3.3968823, 7.9673896, -3.3996463, 7.9685459, -10.2468414, 10.2485924
21: -1.9281859, 8.8792801, -1.9296172, 8.8807945, -8.9053955, 8.9029942
22: -9.2026176, 2.7828126, -9.2039289, 2.7842870, -8.6591797, 8.6716557
23: 1.3826407, 12.5001173, 1.3816026, 12.5067616, -7.7009964, 7.6922569
24: -2.6483424, 10.4836407, -2.6490252, 10.4923306, -8.1243782, 8.1159935
25: 0.4006858, 13.7388496, 0.3986056, 13.7521753, -9.3375874, 9.3262482
26: -17.3781109, 2.4868088, -17.3815117, 2.4874740, -14.5507126, 14.5644455
27: -10.2650509, 6.2909393, -10.2667036, 6.2931690, -9.1003742, 9.0982304
28: 1.1146700, 13.5737991, 1.1132193, 13.5781527, -9.6025276, 9.5943336
29: -5.0815516, 8.3588095, -5.0828571, 8.3604221, -8.5934620, 8.5993729
30: 6.0023270, 17.7053680, 6.0008268, 17.7130241, -7.6404877, 7.6332722
31: -3.3586264, 10.3899660, -3.3599954, 10.3947830, -9.1384964, 9.1319599
32: -19.5718880, -2.7758410, -19.5828094, -2.7753000, -10.6638489, 10.6765575
33: -47.0384636, -21.5756340, -47.0422134, -21.5742149, -14.5144424, 14.5366058
34: -29.7192764, -10.5954046, -29.7242222, -10.5947027, -10.6463470, 10.6563950
35: -29.2250252, -9.9668255, -29.2283287, -9.9659767, -10.6731415, 10.6970367
36: -31.8842335, -9.4217138, -31.8892231, -9.4209156, -12.6694717, 12.6838608
37: -46.1298141, -23.4906807, -46.1343765, -23.4879799, -16.0412064, 16.0670700
38: -34.2988586, -11.5282698, -34.3018112, -11.5256386, -15.1083984, 15.1051788
39: -56.3273926, -30.6880608, -56.3298492, -30.6862392, -13.1686440, 13.1834908
40: -40.2440948, -23.3095722, -40.2511444, -23.3093014, -8.1736717, 8.1882229
41: -26.7252865, -7.0033622, -26.7347603, -7.0026026, -11.2230644, 11.2318878
42: -14.5329266, -2.1249216, -14.5357666, -2.1242659, -8.5039005, 8.5088272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1401

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0890259, upper bound: 5.0831939
time: 14.04 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0890258, upper bound: 5.0831939
time: 46.94 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -20.2244644, 0.6808088, -20.2269707, 0.6877117, -13.4305954, 13.4134483
1: -6.4253831, 5.2793751, -6.4239283, 5.2798462, -6.4025173, 6.3934364
2: -11.0179110, 2.2276411, -11.0238228, 2.2140932, -8.5425606, 8.5687904
3: -12.3466148, 3.3967323, -12.3488503, 3.3876557, -11.1445808, 11.1613197
4: -22.0519905, -5.6344914, -22.0537338, -5.6477842, -9.2486115, 9.2678223
5: -10.7993107, 5.6464353, -10.8071556, 5.6298094, -12.1472778, 12.1774979
6: -22.4874344, -4.4649887, -22.4894314, -4.5091028, -11.1469536, 11.2016602
7: -9.5542946, 8.9105921, -9.5664358, 8.9142170, -12.4549103, 12.4705162
8: -26.3448372, -5.6070957, -26.3501968, -5.6064210, -9.8163223, 9.8175087
9: -14.5926838, 2.1415725, -14.5911722, 2.1394646, -12.9253387, 12.9208603
10: -5.9052544, 11.7238970, -5.9003639, 11.7182159, -13.2210350, 13.2189331
11: 9.5887394, 21.1143723, 9.5996685, 21.1162109, -7.4926319, 7.4739933
12: -15.1516104, 9.8690853, -15.1554623, 9.8685837, -18.4839630, 18.4840469
13: -28.0462074, -3.0750253, -28.0350304, -3.0795622, -12.8849030, 12.8793297
14: -31.3795891, 0.6508820, -31.3577385, 0.6575122, -21.5105057, 21.4740906
15: -24.9269142, -10.7101660, -24.9284840, -10.6949177, -8.8849030, 8.8706703
16: -6.9471054, 7.8909755, -6.9489365, 7.8859038, -10.1169205, 10.1146927
17: -14.7576208, 11.7633553, -14.7414989, 11.7700348, -21.7324066, 21.7046356
18: -0.8861120, 12.5833988, -0.8744671, 12.5816517, -10.6892052, 10.6791954
19: -5.2850142, 4.7419996, -5.2738132, 4.7433271, -7.5707664, 7.5557938
20: -3.4006946, 7.9787803, -3.3981869, 7.9676380, -10.2483788, 10.2632294
21: -1.9411116, 8.8858986, -1.9296405, 8.8817301, -8.9147453, 8.9111824
22: -9.1995296, 2.7668905, -9.1983213, 2.7844677, -8.6724014, 8.6546917
23: 1.3560249, 12.5080652, 1.3819081, 12.5108490, -7.7335625, 7.6988869
24: -2.6863973, 10.4982986, -2.6489780, 10.4994221, -8.1746750, 8.1253433
25: 0.3443089, 13.7586699, 0.4000204, 13.7652130, -9.4118958, 9.3397560
26: -17.3835926, 2.4869256, -17.3764458, 2.4874461, -14.5699768, 14.5570068
27: -10.2676573, 6.2985373, -10.2652321, 6.2934661, -9.1023750, 9.1095428
28: 1.0890520, 13.5824919, 1.1138215, 13.5825844, -9.6266022, 9.6015396
29: -5.0850654, 8.3544941, -5.0802393, 8.3610888, -8.6056156, 8.5953636
30: 5.9772139, 17.7172089, 6.0025759, 17.7198887, -7.6734276, 7.6444969
31: -3.3780913, 10.3938332, -3.3602915, 10.3960228, -9.1581306, 9.1372757
32: -19.5920925, -2.7374485, -19.5927982, -2.7751541, -10.6801414, 10.7300663
33: -47.0305367, -21.5848026, -47.0345955, -21.5747871, -14.5279198, 14.5170059
34: -29.7191925, -10.5942783, -29.7221069, -10.5946617, -10.6499977, 10.6613541
35: -29.2202263, -9.9730997, -29.2232113, -9.9660587, -10.6860886, 10.6844902
36: -31.8817005, -9.4147968, -31.8854618, -9.4206429, -12.6806717, 12.6963081
37: -46.1250610, -23.4990311, -46.1261826, -23.4900436, -16.0709839, 16.0433578
38: -34.3056755, -11.5272942, -34.3033142, -11.5283937, -15.1127930, 15.1136284
39: -56.3200150, -30.6961727, -56.3213959, -30.6872425, -13.1771164, 13.1710777
40: -40.2455368, -23.3011665, -40.2510452, -23.3091888, -8.1808090, 8.1956882
41: -26.7374840, -6.9731297, -26.7404709, -7.0024099, -11.2320747, 11.2676582
42: -14.5351105, -2.1220391, -14.5347729, -2.1242480, -8.5082207, 8.5091171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1401

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0931027, upper bound: 5.0838894
time: 21.82 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0931027, upper bound: 5.0879982
time: 18.51 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -20.2344894, 0.6987231, -20.2274265, 0.6969905, -13.4511490, 13.4171600
1: -6.4306574, 5.2865829, -6.4241877, 5.2839994, -6.4166679, 6.3897724
2: -11.0258999, 2.2402682, -11.0241146, 2.2215874, -8.5616798, 8.5663414
3: -12.3528366, 3.4054475, -12.3492508, 3.3929424, -11.1623383, 11.1583786
4: -22.0551109, -5.6272764, -22.0537071, -5.6438155, -9.2536850, 9.2685356
5: -10.8089037, 5.6621070, -10.8074389, 5.6389680, -12.1694565, 12.1760178
6: -22.4923458, -4.4651175, -22.4910393, -4.5091767, -11.1543579, 11.2025986
7: -9.5695744, 8.9265680, -9.5676699, 8.9237881, -12.4912949, 12.4643974
8: -26.3546753, -5.5978994, -26.3508892, -5.6007528, -9.8445091, 9.8078995
9: -14.5976801, 2.1468194, -14.5941124, 2.1405637, -12.9253235, 12.9313965
10: -5.9093895, 11.7284222, -5.9024925, 11.7193384, -13.2250366, 13.2264442
11: 9.5661898, 21.1191578, 9.5975552, 21.1191177, -7.5171595, 7.4719944
12: -15.1644659, 9.8761492, -15.1626120, 9.8689804, -18.4817047, 18.5087280
13: -28.0459671, -3.0727611, -28.0348434, -3.0780571, -12.8933640, 12.8787384
14: -31.3884277, 0.6647391, -31.3586731, 0.6657457, -21.5274734, 21.4852524
15: -24.9405670, -10.6881895, -24.9366436, -10.6936827, -8.8812904, 8.8950424
16: -6.9517212, 7.8917975, -6.9497857, 7.8860307, -10.1252213, 10.1155434
17: -14.7649240, 11.7764282, -14.7416878, 11.7772999, -21.7415848, 21.7091827
18: -0.8853700, 12.5851154, -0.8750820, 12.5820961, -10.6916885, 10.6813316
19: -5.2882252, 4.7448339, -5.2746029, 4.7444954, -7.5760555, 7.5591278
20: -3.4074161, 7.9807844, -3.3996091, 7.9687815, -10.2593155, 10.2671814
21: -1.9443493, 8.8858385, -1.9306583, 8.8817978, -8.9215355, 8.9110756
22: -9.2105761, 2.7866352, -9.2045078, 2.7853470, -8.6689434, 8.6784248
23: 1.3500434, 12.5123301, 1.3811040, 12.5128288, -7.7407646, 7.7014351
24: -2.6869037, 10.4999256, -2.6493900, 10.4998989, -8.1755600, 8.1280441
25: 0.3403325, 13.7660675, 0.3980155, 13.7658024, -9.4130402, 9.3466721
26: -17.3944435, 2.4934285, -17.3824043, 2.4877489, -14.5699768, 14.5718079
27: -10.2687597, 6.2993431, -10.2662163, 6.2938495, -9.1069756, 9.1109219
28: 1.0857396, 13.5831413, 1.1125317, 13.5826893, -9.6369629, 9.6021729
29: -5.0903940, 8.3639812, -5.0833526, 8.3615608, -8.6044750, 8.6050186
30: 5.9704876, 17.7206306, 5.9999800, 17.7199898, -7.6807518, 7.6455421
31: -3.3822439, 10.3990307, -3.3608785, 10.3984203, -9.1658516, 9.1406174
32: -19.5963478, -2.7371914, -19.5938187, -2.7750034, -10.6841507, 10.7321587
33: -47.0474777, -21.5648632, -47.0440559, -21.5734749, -14.5261574, 14.5493469
34: -29.7279778, -10.5875111, -29.7273674, -10.5942659, -10.6541214, 10.6721611
35: -29.2318115, -9.9601307, -29.2300606, -9.9655886, -10.6799202, 10.7099991
36: -31.8877144, -9.4108686, -31.8891411, -9.4203930, -12.6784439, 12.7061768
37: -46.1409111, -23.4883804, -46.1356163, -23.4893875, -16.0661392, 16.0767288
38: -34.3075447, -11.5217934, -34.3036346, -11.5253811, -15.1179504, 15.1162071
39: -56.3354874, -30.6836815, -56.3311806, -30.6861496, -13.1767082, 13.1901054
40: -40.2565956, -23.2964325, -40.2564468, -23.3091812, -8.1832314, 8.2065773
41: -26.7438736, -6.9711461, -26.7434692, -7.0021405, -11.2388077, 11.2730560
42: -14.5414715, -2.1198831, -14.5371704, -2.1240058, -8.5130806, 8.5155125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1401

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0890259, upper bound: 5.0932014
time: 8.49 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0932015, upper bound: 5.0932014
time: 12.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.88 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 11, lower bound: -5.0931027, upper bound: 5.0739692
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 11, lower bound: -5.0931027, upper bound: 5.0780614
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 22.88
Output dim: 11, lower bound: -5.0890259, upper bound: 5.0831939
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 22.88
Output dim: 11, lower bound: -5.0890258, upper bound: 5.0831939
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 11, lower bound: -5.0931027, upper bound: 5.0838894
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 11, lower bound: -5.0931027, upper bound: 5.0879982
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 11, lower bound: -5.0890259, upper bound: 5.0932014
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 11, lower bound: -5.0932015, upper bound: 5.0932014

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -20.2033062, 0.6591833, -20.2197990, 0.6884534, -13.3890419, 13.3752289
1: -6.4146161, 5.2675390, -6.4219170, 5.2804422, -6.3770924, 6.3740425
2: -10.9967728, 2.1921463, -11.0134754, 2.2126739, -8.5232162, 8.5222187
3: -12.3159666, 3.3627057, -12.3343267, 3.3857050, -11.1126785, 11.1114502
4: -22.0240517, -5.6693673, -22.0391979, -5.6496153, -9.2219696, 9.2168579
5: -10.7727871, 5.6040611, -10.7939081, 5.6281023, -12.1209106, 12.1210175
6: -22.4513321, -4.5187898, -22.4751472, -4.5132098, -11.1022301, 11.1304665
7: -9.5344448, 8.8885460, -9.5577221, 8.9117298, -12.4322205, 12.4412956
8: -26.3179054, -5.6266022, -26.3375549, -5.6072907, -9.7823181, 9.7892723
9: -14.5700817, 2.1139712, -14.5823040, 2.1357183, -12.8964233, 12.8844986
10: -5.8876877, 11.6959877, -5.8965769, 11.7138863, -13.1978645, 13.1847420
11: 9.6401939, 21.0882454, 9.6027832, 21.1028862, -7.4278393, 7.4530640
12: -15.1112928, 9.8243132, -15.1515303, 9.8473043, -18.4195175, 18.4295959
13: -28.0249214, -3.0956421, -28.0331573, -3.0863824, -12.8562851, 12.8517838
14: -31.3147278, 0.6308923, -31.3500519, 0.6489768, -21.4249496, 21.4428177
15: -24.9123287, -10.7210903, -24.9231033, -10.6971588, -8.8594284, 8.8349133
16: -6.9384861, 7.8724146, -6.9463091, 7.8850908, -10.0931320, 10.0896988
17: -14.7008390, 11.7192278, -14.7369518, 11.7458496, -21.6480713, 21.6516571
18: -0.8514779, 12.5624380, -0.8703372, 12.5731258, -10.6409607, 10.6525345
19: -5.2626476, 4.7338281, -5.2712197, 4.7422981, -7.5415936, 7.5409927
20: -3.3788176, 7.9568567, -3.3945892, 7.9664326, -10.2223625, 10.2293663
21: -1.9149482, 8.8734159, -1.9252896, 8.8800669, -8.8848801, 8.8887615
22: -9.1819477, 2.7622523, -9.1961308, 2.7822397, -8.6512451, 8.6381435
23: 1.3986120, 12.4932308, 1.3841941, 12.5046310, -7.6831036, 7.6837234
24: -2.6399071, 10.4790068, -2.6470156, 10.4909973, -8.1116142, 8.1037712
25: 0.4165606, 13.7276878, 0.4028888, 13.7508545, -9.3227272, 9.3110313
26: -17.3429947, 2.4681907, -17.3724346, 2.4803617, -14.5175934, 14.5335617
27: -10.2535334, 6.2880044, -10.2624235, 6.2923994, -9.0802078, 9.0904179
28: 1.1298568, 13.5695992, 1.1168232, 13.5778313, -9.5790749, 9.5865173
29: -5.0610523, 8.3419123, -5.0779085, 8.3564682, -8.5764236, 8.5800056
30: 6.0249195, 17.6963806, 6.0049067, 17.7100048, -7.6223888, 7.6235905
31: -3.3431385, 10.3775330, -3.3548214, 10.3921471, -9.1165352, 9.1123466
32: -19.5600986, -2.7851789, -19.5797195, -2.7773900, -10.6438675, 10.6604919
33: -47.0054970, -21.6223564, -47.0242538, -21.5788498, -14.5002098, 14.4669685
34: -29.6939964, -10.6155605, -29.7104549, -10.5964069, -10.6261749, 10.6188736
35: -29.2051296, -9.9910345, -29.2177238, -9.9680557, -10.6684418, 10.6598320
36: -31.8676147, -9.4271870, -31.8831787, -9.4223690, -12.6537857, 12.6679916
37: -46.1033707, -23.5122910, -46.1224174, -23.4935608, -16.0126266, 16.0141830
38: -34.2836609, -11.5364437, -34.2975960, -11.5296469, -15.0760574, 15.0925751
39: -56.2944946, -30.7184658, -56.3104706, -30.6893501, -13.1563416, 13.1345215
40: -40.2280426, -23.3209457, -40.2434006, -23.3102970, -8.1506310, 8.1656208
41: -26.7121353, -7.0129981, -26.7300053, -7.0044613, -11.1954613, 11.2144547
42: -14.5251265, -2.1335022, -14.5319223, -2.1262352, -8.4741974, 8.4892883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1624

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0845178, upper bound: 5.0728773
time: 20.09 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0920171, upper bound: 5.0728773
time: 22.05 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -20.2124596, 0.6742723, -20.2243767, 0.6889749, -13.3912506, 13.3944016
1: -6.4171948, 5.2734776, -6.4230103, 5.2805924, -6.3777332, 6.3810005
2: -11.0045280, 2.2048283, -11.0178165, 2.2131877, -8.5285606, 8.5384808
3: -12.3300018, 3.3802731, -12.3419485, 3.3864920, -11.1214142, 11.1363106
4: -22.0368195, -5.6549621, -22.0461655, -5.6490250, -9.2307816, 9.2385483
5: -10.7817564, 5.6185589, -10.7986469, 5.6287861, -12.1306076, 12.1408958
6: -22.4596481, -4.5176606, -22.4762039, -4.5141811, -11.1202393, 11.1358643
7: -9.5448170, 8.9008884, -9.5625687, 8.9122620, -12.4418793, 12.4578857
8: -26.3318768, -5.6136522, -26.3451996, -5.6071444, -9.7875023, 9.8094406
9: -14.5818110, 2.1298747, -14.5881214, 2.1370034, -12.9021683, 12.9064178
10: -5.8921938, 11.7093983, -5.8974447, 11.7155199, -13.2023773, 13.2022018
11: 9.6244526, 21.0946808, 9.6017933, 21.1063385, -7.4432993, 7.4514179
12: -15.1389103, 9.8475475, -15.1525097, 9.8602200, -18.4600906, 18.4510040
13: -28.0303383, -3.0903707, -28.0334835, -3.0843325, -12.8626213, 12.8624725
14: -31.3404522, 0.6386278, -31.3529434, 0.6528468, -21.4566040, 21.4435272
15: -24.9093323, -10.7193661, -24.9205589, -10.6963100, -8.8594627, 8.8549061
16: -6.9421263, 7.8826332, -6.9474678, 7.8853745, -10.0938797, 10.1014099
17: -14.7273731, 11.7432938, -14.7377129, 11.7588511, -21.6870651, 21.6691208
18: -0.8712196, 12.5716810, -0.8722270, 12.5778055, -10.6694450, 10.6573982
19: -5.2681217, 4.7385573, -5.2721958, 4.7424402, -7.5463448, 7.5484161
20: -3.3874495, 7.9645319, -3.3965919, 7.9668646, -10.2310677, 10.2419891
21: -1.9219422, 8.8785162, -1.9267483, 8.8802519, -8.8913498, 8.8995361
22: -9.1896257, 2.7622445, -9.1965971, 2.7829170, -8.6598358, 8.6447983
23: 1.3900433, 12.4957151, 1.3832498, 12.5046816, -7.6914406, 7.6866245
24: -2.6463683, 10.4811993, -2.6476982, 10.4913597, -8.1187401, 8.1070385
25: 0.4068201, 13.7308693, 0.4019728, 13.7512178, -9.3331375, 9.3148918
26: -17.3642063, 2.4770799, -17.3737030, 2.4852178, -14.5461998, 14.5396957
27: -10.2612505, 6.2897000, -10.2641068, 6.2924867, -9.0925255, 9.0936241
28: 1.1198754, 13.5728951, 1.1156473, 13.5778980, -9.5892410, 9.5911179
29: -5.0739927, 8.3473759, -5.0783796, 8.3587360, -8.5907707, 8.5808868
30: 6.0104842, 17.6971016, 6.0042648, 17.7098923, -7.6273384, 7.6261711
31: -3.3502741, 10.3845787, -3.3568683, 10.3922729, -9.1235924, 9.1247101
32: -19.5660229, -2.7777731, -19.5807838, -2.7765174, -10.6511421, 10.6686096
33: -47.0181351, -21.5984859, -47.0305176, -21.5773106, -14.4959602, 14.4940910
34: -29.7074127, -10.6032925, -29.7169628, -10.5958395, -10.6231804, 10.6362610
35: -29.2063999, -9.9812241, -29.2171631, -9.9673595, -10.6721497, 10.6650848
36: -31.8757057, -9.4267559, -31.8840809, -9.4218273, -12.6683044, 12.6677628
37: -46.1119995, -23.5053749, -46.1237831, -23.4910030, -16.0432434, 16.0194855
38: -34.2937546, -11.5347786, -34.2995071, -11.5292912, -15.0996933, 15.0940323
39: -56.3084641, -30.7024231, -56.3179054, -30.6884766, -13.1519203, 13.1552162
40: -40.2314911, -23.3154564, -40.2448120, -23.3099613, -8.1665039, 8.1652603
41: -26.7175503, -7.0071774, -26.7309837, -7.0039024, -11.2106438, 11.2163162
42: -14.5254774, -2.1286831, -14.5327148, -2.1254981, -8.4948044, 8.4894753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1624

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0845178, upper bound: 5.0769718
time: 23.76 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0920171, upper bound: 5.0769718
time: 17.67 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -20.2118530, 0.6645143, -20.2202587, 0.6864617, -13.4165649, 13.3901558
1: -6.4216347, 5.2725563, -6.4221382, 5.2791786, -6.3974762, 6.3844471
2: -11.0074615, 2.2138653, -11.0178833, 2.2129085, -8.5313263, 8.5490913
3: -12.3248100, 3.3775444, -12.3365459, 3.3859234, -11.1213608, 11.1301804
4: -22.0334358, -5.6503029, -22.0431938, -5.6492219, -9.2285805, 9.2406387
5: -10.7853680, 5.6308651, -10.7993565, 5.6285191, -12.1310654, 12.1527901
6: -22.4777756, -4.4743490, -22.4876175, -4.5131788, -11.1229668, 11.1875420
7: -9.5398731, 8.8967152, -9.5591526, 8.9128475, -12.4387207, 12.4490662
8: -26.3249855, -5.6207070, -26.3389511, -5.6069527, -9.7957840, 9.7921944
9: -14.5766230, 2.1221519, -14.5827351, 2.1359851, -12.9064178, 12.8934937
10: -5.8968329, 11.7073803, -5.8971848, 11.7147055, -13.2094536, 13.1963120
11: 9.6059818, 21.1056366, 9.6015625, 21.1112232, -7.4718723, 7.4668865
12: -15.1215811, 9.8327599, -15.1529818, 9.8478060, -18.4323730, 18.4441986
13: -28.0392532, -3.0831518, -28.0337639, -3.0833631, -12.8750687, 12.8647079
14: -31.3475266, 0.6410155, -31.3510170, 0.6522756, -21.4722061, 21.4589462
15: -24.9174271, -10.7130671, -24.9233704, -10.6964550, -8.8684692, 8.8441315
16: -6.9420958, 7.8788705, -6.9469643, 7.8844185, -10.1081314, 10.0986710
17: -14.7261677, 11.7269554, -14.7377014, 11.7490416, -21.6805038, 21.6633224
18: -0.8632815, 12.5690746, -0.8707626, 12.5738373, -10.6559715, 10.6604195
19: -5.2779474, 4.7370305, -5.2718697, 4.7430277, -7.5607166, 7.5454884
20: -3.3893509, 7.9702735, -3.3945575, 7.9666781, -10.2348557, 10.2479515
21: -1.9311177, 8.8799858, -1.9263582, 8.8810482, -8.9010162, 8.8968353
22: -9.1899033, 2.7660646, -9.1966972, 2.7833076, -8.6610260, 8.6449051
23: 1.3659984, 12.5054512, 1.3836918, 12.5107088, -7.7228775, 7.6929054
24: -2.6784611, 10.4953251, -2.6473985, 10.4985428, -8.1627960, 8.1158218
25: 0.3561990, 13.7548952, 0.4022374, 13.7645016, -9.3981686, 9.3314552
26: -17.3593369, 2.4747908, -17.3733768, 2.4806376, -14.5368576, 14.5409088
27: -10.2572842, 6.2964358, -10.2619505, 6.2931252, -9.0868130, 9.1031380
28: 1.1009207, 13.5789433, 1.1161473, 13.5823603, -9.6135139, 9.5943565
29: -5.0698967, 8.3471022, -5.0783982, 8.3576012, -8.5874367, 8.5856667
30: 5.9930649, 17.7116013, 6.0040655, 17.7169552, -7.6626282, 7.6358604
31: -3.3667789, 10.3865938, -3.3557196, 10.3957577, -9.1439056, 9.1210194
32: -19.5845470, -2.7464907, -19.5907459, -2.7770751, -10.6641731, 10.7160892
33: -47.0145454, -21.6116333, -47.0260773, -21.5781536, -14.5119247, 14.4797058
34: -29.7026863, -10.6076908, -29.7136116, -10.5959530, -10.6339836, 10.6346359
35: -29.2118912, -9.9843311, -29.2194481, -9.9676685, -10.6752090, 10.6728020
36: -31.8711357, -9.4163275, -31.8830204, -9.4218445, -12.6627769, 12.6902809
37: -46.1145096, -23.5099506, -46.1235695, -23.4950256, -16.0375214, 16.0238113
38: -34.2923622, -11.5299377, -34.2994385, -11.5293884, -15.0856018, 15.1036415
39: -56.3024979, -30.7140961, -56.3117867, -30.6892128, -13.1643829, 13.1411629
40: -40.2405281, -23.3078022, -40.2487183, -23.3101845, -8.1601830, 8.1839714
41: -26.7306824, -6.9807291, -26.7387085, -7.0040684, -11.2111893, 11.2556000
42: -14.5336733, -2.1284525, -14.5333395, -2.1259675, -8.4833736, 8.4959717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1624

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0845178, upper bound: 5.0827944
time: 20.18 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0921149, upper bound: 5.0827944
time: 17.10 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -20.2209854, 0.6795678, -20.2248840, 0.6869493, -13.4187508, 13.4092865
1: -6.4241991, 5.2785120, -6.4232306, 5.2793298, -6.3981209, 6.3914146
2: -11.0151863, 2.2265530, -11.0222216, 2.2134182, -8.5366631, 8.5653610
3: -12.3388271, 3.3950930, -12.3441620, 3.3866863, -11.1301041, 11.1550293
4: -22.0462475, -5.6358604, -22.0501442, -5.6486034, -9.2373810, 9.2623138
5: -10.7942600, 5.6453533, -10.8040848, 5.6291828, -12.1407928, 12.1726685
6: -22.4860744, -4.4732199, -22.4886055, -4.5141640, -11.1409607, 11.1929283
7: -9.5502605, 8.9091053, -9.5640011, 8.9133453, -12.4483719, 12.4656639
8: -26.3389816, -5.6077900, -26.3466225, -5.6068382, -9.8009872, 9.8123589
9: -14.5883570, 2.1380439, -14.5885057, 2.1373348, -12.9121933, 12.9154434
10: -5.9013762, 11.7208185, -5.8980141, 11.7163315, -13.2139130, 13.2137642
11: 9.5902405, 21.1120815, 9.6006012, 21.1146736, -7.4873409, 7.4652271
12: -15.1491804, 9.8560457, -15.1539936, 9.8607206, -18.4729691, 18.4655838
13: -28.0446301, -3.0779634, -28.0340767, -3.0813005, -12.8813934, 12.8753853
14: -31.3732109, 0.6486766, -31.3538933, 0.6560726, -21.5038071, 21.4596329
15: -24.9144306, -10.7113113, -24.9208336, -10.6955814, -8.8684921, 8.8641186
16: -6.9457226, 7.8890572, -6.9481311, 7.8847179, -10.1088791, 10.1103745
17: -14.7527180, 11.7510624, -14.7385082, 11.7619905, -21.7195435, 21.6807709
18: -0.8830230, 12.5783195, -0.8726175, 12.5785141, -10.6844177, 10.6652603
19: -5.2834282, 4.7417474, -5.2728386, 4.7431660, -7.5654697, 7.5528965
20: -3.3979905, 7.9779520, -3.3965521, 7.9671063, -10.2435341, 10.2605820
21: -1.9381413, 8.8850927, -1.9277983, 8.8812313, -8.9075012, 8.9076118
22: -9.1976023, 2.7660902, -9.1971321, 2.7839832, -8.6695862, 8.6515446
23: 1.3574264, 12.5079308, 1.3827718, 12.5107517, -7.7312164, 7.6958103
24: -2.6849020, 10.4974937, -2.6480627, 10.4989271, -8.1699295, 8.1190987
25: 0.3464837, 13.7580833, 0.4013393, 13.7648602, -9.4085770, 9.3353004
26: -17.3805542, 2.4837096, -17.3745918, 2.4855068, -14.5654602, 14.5470428
27: -10.2649536, 6.2980971, -10.2636003, 6.2932343, -9.0991402, 9.1063290
28: 1.0909560, 13.5822315, 1.1149728, 13.5824432, -9.6236877, 9.5989647
29: -5.0828257, 8.3525562, -5.0788813, 8.3598938, -8.6017838, 8.5865555
30: 5.9786377, 17.7123489, 6.0034122, 17.7168579, -7.6675930, 7.6384411
31: -3.3738818, 10.3936405, -3.3577652, 10.3958912, -9.1509590, 9.1333694
32: -19.5905190, -2.7391174, -19.5918503, -2.7762008, -10.6714478, 10.7241993
33: -47.0271301, -21.5876942, -47.0323792, -21.5765839, -14.5076599, 14.5068207
34: -29.7161255, -10.5954494, -29.7200737, -10.5953999, -10.6309662, 10.6520119
35: -29.2132206, -9.9745407, -29.2189140, -9.9669456, -10.6789131, 10.6780586
36: -31.8792305, -9.4158611, -31.8839836, -9.4212599, -12.6772957, 12.6900711
37: -46.1231117, -23.5030251, -46.1249695, -23.4925060, -16.0681305, 16.0291519
38: -34.3024635, -11.5282421, -34.3012733, -11.5290184, -15.1092377, 15.1050949
39: -56.3165092, -30.6980572, -56.3191910, -30.6883545, -13.1599503, 13.1618652
40: -40.2439804, -23.3022995, -40.2501068, -23.3098927, -8.1760521, 8.1836071
41: -26.7361240, -6.9749393, -26.7396526, -7.0034862, -11.2263870, 11.2574921
42: -14.5340443, -2.1236486, -14.5341301, -2.1252370, -8.5040016, 8.4961662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 663

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1624

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0845178, upper bound: 5.0869027
time: 22.37 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0921149, upper bound: 5.0869027
time: 23.91 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -20.2277145, 0.6974032, -20.2148170, 0.6805894, -13.4278488, 13.4031181
1: -6.4288497, 5.2858853, -6.4204469, 5.2771988, -6.4076920, 6.3847218
2: -11.0199537, 2.2390800, -11.0136757, 2.2077928, -8.5419769, 8.5550957
3: -12.3405304, 3.4037108, -12.3274307, 3.3737679, -11.1311569, 11.1351433
4: -22.0445442, -5.6287594, -22.0352249, -5.6596742, -9.2264748, 9.2485313
5: -10.8011141, 5.6607847, -10.7934647, 5.6233940, -12.1447449, 12.1597672
6: -22.4905434, -4.4691849, -22.4813709, -4.5185232, -11.1402359, 11.1786270
7: -9.5623293, 8.9251804, -9.5532608, 8.9099436, -12.4698639, 12.4482079
8: -26.3434715, -5.5984359, -26.3310146, -5.6143346, -9.8192024, 9.7873611
9: -14.5892372, 2.1433942, -14.5780563, 2.1211631, -12.8979721, 12.9124947
10: -5.9061747, 11.7248821, -5.8940558, 11.7028112, -13.2024155, 13.2148666
11: 9.5680847, 21.1141853, 9.6148090, 21.1104164, -7.5100660, 7.4512310
12: -15.1619806, 9.8553505, -15.1325693, 9.8326559, -18.4418335, 18.4571533
13: -28.0446663, -3.0765679, -28.0278339, -3.0861475, -12.8787079, 12.8689156
14: -31.3816643, 0.6595092, -31.3265896, 0.6558058, -21.5123215, 21.4469223
15: -24.9354591, -10.6897221, -24.9271965, -10.6965723, -8.8547516, 8.8785954
16: -6.9497333, 7.8903141, -6.9448018, 7.8738813, -10.1091881, 10.1067276
17: -14.7611542, 11.7554512, -14.7102585, 11.7408772, -21.7002869, 21.6573257
18: -0.8816459, 12.5773058, -0.8522670, 12.5677719, -10.6729202, 10.6480827
19: -5.2862673, 4.7445402, -5.2675180, 4.7395139, -7.5657501, 7.5490742
20: -3.4037886, 7.9798441, -3.3882434, 7.9602609, -10.2440414, 10.2536125
21: -1.9410782, 8.8851700, -1.9206796, 8.8758583, -8.9071960, 8.8973427
22: -9.2089500, 2.7854757, -9.1948652, 2.7845521, -8.6591492, 8.6670418
23: 1.3518317, 12.5121632, 1.3910763, 12.5102158, -7.7347908, 7.6907597
24: -2.6853192, 10.4990635, -2.6414506, 10.4968643, -8.1660271, 8.1161766
25: 0.3425694, 13.7653522, 0.4099278, 13.7620354, -9.4047203, 9.3329391
26: -17.3913918, 2.4866328, -17.3581676, 2.4756157, -14.5538483, 14.5386963
27: -10.2654362, 6.2989855, -10.2558088, 6.2917576, -9.1005630, 9.0953674
28: 1.0880680, 13.5829258, 1.1243858, 13.5791302, -9.6297989, 9.5890884
29: -5.0885744, 8.3605595, -5.0681810, 8.3541708, -8.5947647, 8.5868607
30: 5.9719715, 17.7176781, 6.0158334, 17.7143974, -7.6721230, 7.6347542
31: -3.3776624, 10.3987732, -3.3495650, 10.3911505, -9.1496201, 9.1263771
32: -19.5942879, -2.7391093, -19.5862961, -2.7840638, -10.6702042, 10.7161903
33: -47.0390701, -21.5682793, -47.0280876, -21.6003780, -14.4888687, 14.5333138
34: -29.7194691, -10.5888042, -29.7108116, -10.6076794, -10.6273804, 10.6561279
35: -29.2280216, -9.9617138, -29.2217560, -9.9768200, -10.6682434, 10.6991119
36: -31.8852482, -9.4121037, -31.8785973, -9.4218788, -12.6724205, 12.6882362
37: -46.1383133, -23.4933434, -46.1250267, -23.5003319, -16.0465775, 16.0432816
38: -34.3036728, -11.5227375, -34.2903557, -11.5280771, -15.1079788, 15.0890121
39: -56.3258514, -30.6857224, -56.3137436, -30.7040539, -13.1467934, 13.1774101
40: -40.2542191, -23.2974586, -40.2514114, -23.3158035, -8.1715164, 8.1859474
41: -26.7420578, -6.9727554, -26.7366867, -7.0097361, -11.2267723, 11.2521706
42: -14.5400429, -2.1216047, -14.5357542, -2.1304190, -8.4999371, 8.4906864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1624

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0879377, upper bound: 5.0846167
time: 23.40 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0879377, upper bound: 5.0921147
time: 27.90 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -20.2323647, 0.6978910, -20.2239532, 0.6956887, -13.4470062, 13.4053192
1: -6.4299464, 5.2860808, -6.4230175, 5.2831240, -6.4146538, 6.3853779
2: -11.0242624, 2.2396057, -11.0214176, 2.2204747, -8.5582428, 8.5604401
3: -12.3481426, 3.4044714, -12.3414536, 3.3913050, -11.1560440, 11.1438599
4: -22.0515480, -5.6281624, -22.0479698, -5.6452398, -9.2481537, 9.2573204
5: -10.8058462, 5.6614475, -10.8023777, 5.6379032, -12.1646271, 12.1695099
6: -22.4915371, -4.4701672, -22.4896717, -4.5174203, -11.1456299, 11.1966095
7: -9.5671749, 8.9256430, -9.5636654, 8.9222813, -12.4864655, 12.4578705
8: -26.3511372, -5.5982575, -26.3449821, -5.6013980, -9.8393898, 9.7925644
9: -14.5949841, 2.1446903, -14.5897655, 2.1370487, -12.9199371, 12.9182816
10: -5.9070344, 11.7265301, -5.8986149, 11.7162495, -13.2198677, 13.2193451
11: 9.5670967, 21.1176472, 9.5990543, 21.1168404, -7.5084009, 7.4666901
12: -15.1630230, 9.8682537, -15.1601591, 9.8558979, -18.4632645, 18.4977264
13: -28.0449924, -3.0745149, -28.0332184, -3.0809455, -12.8894043, 12.8752632
14: -31.3845596, 0.6633866, -31.3522377, 0.6635053, -21.5130081, 21.4785385
15: -24.9329453, -10.6888847, -24.9242077, -10.6948128, -8.8747406, 8.8786354
16: -6.9508867, 7.8906264, -6.9484453, 7.8840961, -10.1208992, 10.1074829
17: -14.7619534, 11.7684622, -14.7367792, 11.7649498, -21.7177277, 21.6963348
18: -0.8835268, 12.5819702, -0.8720124, 12.5770159, -10.6777611, 10.6765366
19: -5.2872343, 4.7446733, -5.2729902, 4.7442427, -7.5731411, 7.5538292
20: -3.4057758, 7.9802618, -3.3968844, 7.9679341, -10.2566605, 10.2623215
21: -1.9425492, 8.8853321, -1.9276729, 8.8809786, -8.9179688, 8.9038200
22: -9.2094307, 2.7861333, -9.2025690, 2.7845471, -8.6657982, 8.6756115
23: 1.3508791, 12.5122318, 1.3825206, 12.5126829, -7.7376842, 7.6990948
24: -2.6859932, 10.4994469, -2.6479068, 10.4990559, -8.1693211, 8.1232910
25: 0.3416526, 13.7657022, 0.4001834, 13.7652168, -9.4085808, 9.3433380
26: -17.3926353, 2.4914808, -17.3793869, 2.4845333, -14.5600014, 14.5672913
27: -10.2671585, 6.2990966, -10.2635231, 6.2934017, -9.1037750, 9.1076927
28: 1.0868900, 13.5829849, 1.1144021, 13.5824413, -9.6343880, 9.5992699
29: -5.0890265, 8.3628216, -5.0811162, 8.3596268, -8.5956402, 8.6012001
30: 5.9713397, 17.7175865, 6.0013981, 17.7151184, -7.6746979, 7.6397038
31: -3.3796883, 10.3989248, -3.3566511, 10.3982038, -9.1619797, 9.1334515
32: -19.5953693, -2.7382188, -19.5922279, -2.7766881, -10.6783066, 10.7234554
33: -47.0452919, -21.5666962, -47.0406647, -21.5764923, -14.5159874, 14.5290794
34: -29.7259369, -10.5882416, -29.7242432, -10.5954676, -10.6447830, 10.6531219
35: -29.2275276, -9.9610195, -29.2230873, -9.9670534, -10.6734962, 10.7028008
36: -31.8862305, -9.4114847, -31.8866730, -9.4214430, -12.6722031, 12.7027779
37: -46.1397247, -23.4907875, -46.1336212, -23.4934387, -16.0519180, 16.0738297
38: -34.3056145, -11.5223866, -34.3004608, -11.5263796, -15.1094208, 15.1126595
39: -56.3333130, -30.6848698, -56.3276901, -30.6880207, -13.1674728, 13.1729927
40: -40.2556152, -23.2971363, -40.2548828, -23.3102970, -8.1711464, 8.2018242
41: -26.7430458, -6.9721937, -26.7421055, -7.0039225, -11.2286205, 11.2673187
42: -14.5408611, -2.1208751, -14.5361204, -2.1256337, -8.5001202, 8.5112896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1624

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0921149, upper bound: 5.0846167
time: 18.45 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0921149, upper bound: 5.0921147
time: 18.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 38.74 seconds
IS_A1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0845178, upper bound: 5.0728773
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0920171, upper bound: 5.0728773
IS_A1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0845178, upper bound: 5.0769718
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0920171, upper bound: 5.0769718
IS_A2_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0845178, upper bound: 5.0827944
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0921149, upper bound: 5.0827944
IS_A2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0845178, upper bound: 5.0869027
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0921149, upper bound: 5.0869027
IS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0879377, upper bound: 5.0846167
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0879377, upper bound: 5.0921147
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0921149, upper bound: 5.0846167
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 38.74
Output dim: 11, lower bound: -5.0921149, upper bound: 5.0921147

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -20.2023849, 0.6588054, -20.2212963, 0.7037048, -13.4037628, 13.3784218
1: -6.4144826, 5.2667875, -6.4272685, 5.2835007, -6.3836842, 6.3759460
2: -10.9967117, 2.1919277, -11.0170164, 2.2247517, -8.5465546, 8.5165100
3: -12.3146515, 3.3624661, -12.3342609, 3.4206243, -11.1463280, 11.1052780
4: -22.0237312, -5.6695409, -22.0402565, -5.6398563, -9.2403908, 9.2076149
5: -10.7722950, 5.6038265, -10.7943182, 5.6496320, -12.1482849, 12.1163826
6: -22.4504967, -4.5188570, -22.4808006, -4.4931602, -11.1213799, 11.1384506
7: -9.5339212, 8.8883896, -9.5596294, 8.9205790, -12.4597855, 12.4238930
8: -26.3176689, -5.6280079, -26.3494110, -5.6072245, -9.7894058, 9.7975998
9: -14.5679493, 2.1135297, -14.5828514, 2.1894138, -12.9528580, 12.8712044
10: -5.8851089, 11.6955385, -5.8965740, 11.7509642, -13.2393570, 13.1857567
11: 9.6406994, 21.0861950, 9.5758152, 21.1007004, -7.4177704, 7.4788132
12: -15.1108656, 9.8240337, -15.1607523, 9.8507891, -18.3920975, 18.4693604
13: -28.0217667, -3.0961890, -28.0303860, -3.0141573, -12.9326324, 12.8340530
14: -31.3140068, 0.6294689, -31.3995132, 0.6506963, -21.4234085, 21.4931183
15: -24.9110737, -10.7215757, -24.9218273, -10.6867752, -8.8706779, 8.8364792
16: -6.9370494, 7.8722782, -6.9470029, 7.9088578, -10.1159821, 10.0894661
17: -14.7004833, 11.7167759, -14.7897968, 11.7434120, -21.6428299, 21.7106476
18: -0.8508816, 12.5601025, -0.9302130, 12.5716667, -10.6250954, 10.7140121
19: -5.2624450, 4.7327003, -5.3014269, 4.7413926, -7.5370064, 7.5693817
20: -3.3782704, 7.9567747, -3.4054458, 7.9705462, -10.2441864, 10.2240067
21: -1.9145968, 8.8726473, -1.9477401, 8.8795738, -8.8979225, 8.9030533
22: -9.1817436, 2.7615199, -9.2210751, 2.7827578, -8.6520481, 8.6624756
23: 1.3987935, 12.4911165, 1.3568586, 12.5029087, -7.6781807, 7.7127781
24: -2.6396973, 10.4770451, -2.6834836, 10.4915257, -8.1060524, 8.1410637
25: 0.4168298, 13.7266655, 0.3825288, 13.7516785, -9.3227234, 9.3317184
26: -17.3424950, 2.4670310, -17.4230118, 2.4796290, -14.5092430, 14.5820160
27: -10.2530184, 6.2859759, -10.3226595, 6.2901525, -9.0804329, 9.1512051
28: 1.1300921, 13.5684013, 1.0784945, 13.5769491, -9.5777168, 9.6249542
29: -5.0608869, 8.3402805, -5.1171417, 8.3557711, -8.5728703, 8.6215286
30: 6.0251751, 17.6951351, 5.9876280, 17.7093334, -7.6251106, 7.6353760
31: -3.3428161, 10.3761845, -3.3860295, 10.3905888, -9.1197853, 9.1407051
32: -19.5579529, -2.7853510, -19.5858803, -2.7434165, -10.6771927, 10.6630440
33: -47.0019073, -21.6229019, -47.0216560, -21.5155602, -14.5673370, 14.4575500
34: -29.6914310, -10.6157150, -29.7133904, -10.5718727, -10.6442795, 10.6412315
35: -29.2025566, -9.9913807, -29.2183266, -9.9401093, -10.6773415, 10.6784859
36: -31.8656864, -9.4273243, -31.8912506, -9.3981628, -12.6702271, 12.6959991
37: -46.1027069, -23.5126686, -46.1259499, -23.4793034, -16.0000610, 16.0416412
38: -34.2828369, -11.5365477, -34.3050003, -11.5170612, -15.0786133, 15.1196518
39: -56.2893448, -30.7186527, -56.3050919, -30.6321259, -13.2175941, 13.1186905
40: -40.2258759, -23.3209705, -40.2425003, -23.2929688, -8.1668701, 8.1611862
41: -26.7115479, -7.0131168, -26.7381325, -6.9884710, -11.2094612, 11.2348785
42: -14.5239410, -2.1336434, -14.5361071, -2.0978923, -8.4967289, 8.4938793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0827143, upper bound: 5.0724041
time: 15.77 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0752063, upper bound: 5.0724041
time: 18.64 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -20.2115479, 0.6739593, -20.2259407, 0.7042336, -13.4059906, 13.3975716
1: -6.4170499, 5.2727489, -6.4283581, 5.2836571, -6.3843460, 6.3829250
2: -11.0044518, 2.2046287, -11.0213528, 2.2252543, -8.5518951, 8.5327816
3: -12.3287001, 3.3799963, -12.3418293, 3.4213681, -11.1550674, 11.1301460
4: -22.0365181, -5.6551452, -22.0472240, -5.6393061, -9.2491837, 9.2293129
5: -10.7812338, 5.6183252, -10.7990961, 5.6503043, -12.1580429, 12.1362953
6: -22.4587936, -4.5177231, -22.4817944, -4.4941201, -11.1393661, 11.1438675
7: -9.5442839, 8.9007235, -9.5644550, 8.9210625, -12.4694366, 12.4405060
8: -26.3316784, -5.6150637, -26.3570538, -5.6070752, -9.7945938, 9.8178024
9: -14.5796509, 2.1294458, -14.5886307, 2.1906974, -12.9585876, 12.8931427
10: -5.8896322, 11.7089596, -5.8974237, 11.7526302, -13.2438202, 13.2031975
11: 9.6249609, 21.0926323, 9.5748405, 21.1041374, -7.4332447, 7.4771366
12: -15.1384659, 9.8473196, -15.1617069, 9.8636723, -18.4327011, 18.4907761
13: -28.0271168, -3.0909166, -28.0307350, -3.0121267, -12.9389648, 12.8447189
14: -31.3396759, 0.6371899, -31.4024162, 0.6545105, -21.4549942, 21.4938583
15: -24.9080544, -10.7198439, -24.9193115, -10.6859198, -8.8707047, 8.8564739
16: -6.9406672, 7.8824630, -6.9481621, 7.9091535, -10.1167450, 10.1011753
17: -14.7269735, 11.7408752, -14.7905912, 11.7563591, -21.6818924, 21.7280807
18: -0.8706043, 12.5693521, -0.9320998, 12.5763044, -10.6535454, 10.7188606
19: -5.2679510, 4.7374239, -5.3024368, 4.7415295, -7.5417557, 7.5766468
20: -3.3868916, 7.9644337, -3.4074411, 7.9709940, -10.2528839, 10.2366409
21: -1.9216046, 8.8777618, -1.9491972, 8.8797579, -8.9043808, 8.9138527
22: -9.1894388, 2.7615533, -9.2215023, 2.7834461, -8.6606121, 8.6691246
23: 1.3902458, 12.4935741, 1.3559386, 12.5029526, -7.6865253, 7.7156258
24: -2.6461749, 10.4792128, -2.6841531, 10.4918976, -8.1131611, 8.1443329
25: 0.4070940, 13.7298517, 0.3816063, 13.7520466, -9.3331223, 9.3355713
26: -17.3637295, 2.4759617, -17.4242573, 2.4844842, -14.5378113, 14.5881348
27: -10.2607174, 6.2876711, -10.3243446, 6.2902479, -9.0927505, 9.1544209
28: 1.1201441, 13.5717049, 1.0773060, 13.5769873, -9.5878906, 9.6295471
29: -5.0737972, 8.3457260, -5.1175771, 8.3580399, -8.5871964, 8.6224556
30: 6.0107594, 17.6958694, 5.9869928, 17.7092419, -7.6300678, 7.6378880
31: -3.3498938, 10.3832397, -3.3879488, 10.3907223, -9.1268311, 9.1530647
32: -19.5639076, -2.7779675, -19.5869675, -2.7425082, -10.6844673, 10.6710987
33: -47.0145073, -21.5990524, -47.0279388, -21.5139523, -14.5630913, 14.4846039
34: -29.7048664, -10.6034546, -29.7198715, -10.5713043, -10.6412926, 10.6586037
35: -29.2038727, -9.9815578, -29.2178154, -9.9394341, -10.6810036, 10.6837311
36: -31.8737679, -9.4268732, -31.8922081, -9.3975878, -12.6847038, 12.6957588
37: -46.1113548, -23.5057030, -46.1273613, -23.4767933, -16.0306549, 16.0469284
38: -34.2930031, -11.5348482, -34.3069229, -11.5167084, -15.1022339, 15.1211357
39: -56.3033104, -30.7026558, -56.3125114, -30.6312675, -13.2132339, 13.1393852
40: -40.2293777, -23.3154964, -40.2438812, -23.2926369, -8.1826630, 8.1608162
41: -26.7169399, -7.0073009, -26.7391167, -6.9879036, -11.2245312, 11.2367287
42: -14.5243015, -2.1288273, -14.5368996, -2.0971441, -8.5172768, 8.4940586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0827143, upper bound: 5.0764985
time: 17.16 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0915595, upper bound: 5.0764985
time: 19.18 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -20.2109165, 0.6640821, -20.2217426, 0.7016838, -13.4313087, 13.3933258
1: -6.4215078, 5.2718134, -6.4274611, 5.2822261, -6.4040680, 6.3863583
2: -11.0073776, 2.2136497, -11.0214252, 2.2249982, -8.5546494, 8.5433769
3: -12.3234940, 3.3772814, -12.3364506, 3.4208288, -11.1550140, 11.1239815
4: -22.0331497, -5.6504927, -22.0442657, -5.6395116, -9.2470016, 9.2313995
5: -10.7848043, 5.6306071, -10.7997837, 5.6500311, -12.1585007, 12.1481361
6: -22.4769459, -4.4744344, -22.4932365, -4.4931045, -11.1421242, 11.1955261
7: -9.5393066, 8.8965712, -9.5610447, 8.9216938, -12.4662552, 12.4316559
8: -26.3247643, -5.6221027, -26.3508377, -5.6069317, -9.8028793, 9.8005104
9: -14.5744877, 2.1217129, -14.5832443, 2.1897244, -12.9628906, 12.8802414
10: -5.8942575, 11.7069187, -5.8971424, 11.7517633, -13.2508850, 13.1973114
11: 9.6065016, 21.1036186, 9.5745897, 21.1090279, -7.4618053, 7.4926167
12: -15.1210957, 9.8325081, -15.1621857, 9.8513060, -18.4050064, 18.4839630
13: -28.0361290, -3.0837216, -28.0309811, -3.0111520, -12.9514389, 12.8469543
14: -31.3467102, 0.6395447, -31.4004459, 0.6538703, -21.4706116, 21.5092316
15: -24.9161491, -10.7135534, -24.9221096, -10.6860371, -8.8796921, 8.8456917
16: -6.9406409, 7.8786926, -6.9476633, 7.9081535, -10.1309586, 10.0984077
17: -14.7258062, 11.7245178, -14.7906055, 11.7465582, -21.6753693, 21.7222519
18: -0.8626604, 12.5667238, -0.9306264, 12.5723696, -10.6400681, 10.7218857
19: -5.2777495, 4.7358828, -5.3020697, 4.7421222, -7.5561333, 7.5738621
20: -3.3887842, 7.9701724, -3.4054244, 7.9708009, -10.2566528, 10.2425728
21: -1.9307561, 8.8792095, -1.9487847, 8.8805771, -8.9140587, 8.9111328
22: -9.1897011, 2.7653663, -9.2216291, 2.7838407, -8.6618023, 8.6692295
23: 1.3662007, 12.5033064, 1.3564132, 12.5089855, -7.7179470, 7.7219563
24: -2.6782842, 10.4933243, -2.6838324, 10.4990730, -8.1572227, 8.1531181
25: 0.3564694, 13.7538776, 0.3819168, 13.7653294, -9.3981571, 9.3521194
26: -17.3588963, 2.4736516, -17.4239464, 2.4799194, -14.5284691, 14.5893478
27: -10.2567329, 6.2943835, -10.3222132, 6.2908478, -9.0870323, 9.1639175
28: 1.1011705, 13.5777464, 1.0777926, 13.5814400, -9.6121597, 9.6328049
29: -5.0697231, 8.3454695, -5.1176305, 8.3569183, -8.5838757, 8.6271973
30: 5.9933348, 17.7103882, 5.9867682, 17.7163162, -7.6653481, 7.6476364
31: -3.3664415, 10.3852749, -3.3869181, 10.3941774, -9.1471405, 9.1493626
32: -19.5824242, -2.7466636, -19.5969143, -2.7431192, -10.6974983, 10.7186470
33: -47.0109177, -21.6121712, -47.0235558, -21.5147991, -14.5790596, 14.4702873
34: -29.7001534, -10.6078453, -29.7165565, -10.5714483, -10.6520729, 10.6569710
35: -29.2093391, -9.9846878, -29.2200718, -9.9397202, -10.6841164, 10.6914330
36: -31.8692245, -9.4164810, -31.8911285, -9.3976059, -12.6792183, 12.7182961
37: -46.1138153, -23.5103188, -46.1271629, -23.4807739, -16.0249634, 16.0512772
38: -34.2914848, -11.5300274, -34.3068237, -11.5167885, -15.0881729, 15.1307259
39: -56.2973328, -30.7143478, -56.3064079, -30.6319447, -13.2256546, 13.1253090
40: -40.2383499, -23.3078365, -40.2477837, -23.2928791, -8.1764107, 8.1795387
41: -26.7301102, -6.9808908, -26.7468452, -6.9880686, -11.2251968, 11.2760429
42: -14.5324764, -2.1285901, -14.5374851, -2.0976396, -8.5059185, 8.5005569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0827451, upper bound: 5.0822870
time: 24.43 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0916038, upper bound: 5.0822870
time: 22.86 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -20.2200584, 0.6792402, -20.2263813, 0.7022157, -13.4334869, 13.4124832
1: -6.4240785, 5.2777596, -6.4285645, 5.2824001, -6.4047089, 6.3933201
2: -11.0151081, 2.2263246, -11.0257645, 2.2255075, -8.5599861, 8.5596542
3: -12.3374929, 3.3948383, -12.3440742, 3.4215984, -11.1637764, 11.1488609
4: -22.0459614, -5.6360645, -22.0512257, -5.6389036, -9.2557755, 9.2530823
5: -10.7937422, 5.6451240, -10.8045092, 5.6506777, -12.1682205, 12.1680450
6: -22.4852676, -4.4733047, -22.4942455, -4.4940786, -11.1600952, 11.2009277
7: -9.5497017, 8.9089432, -9.5658808, 8.9221287, -12.4759216, 12.4483032
8: -26.3387661, -5.6092033, -26.3585377, -5.6067667, -9.8080788, 9.8207092
9: -14.5862179, 2.1376245, -14.5890312, 2.1910267, -12.9686432, 12.9021568
10: -5.8987913, 11.7203770, -5.8979874, 11.7534075, -13.2554016, 13.2147598
11: 9.5907574, 21.1100426, 9.5736446, 21.1124706, -7.4772816, 7.4909592
12: -15.1487017, 9.8557997, -15.1632433, 9.8641825, -18.4455795, 18.5053635
13: -28.0415325, -3.0784936, -28.0313263, -3.0090885, -12.9577255, 12.8576508
14: -31.3724022, 0.6472378, -31.4033241, 0.6577952, -21.5022583, 21.5099945
15: -24.9131508, -10.7117958, -24.9195709, -10.6851959, -8.8797340, 8.8656960
16: -6.9442854, 7.8888884, -6.9488096, 7.9084425, -10.1316986, 10.1101322
17: -14.7523232, 11.7485981, -14.7913656, 11.7595291, -21.7143784, 21.7397537
18: -0.8824079, 12.5759602, -0.9324992, 12.5770359, -10.6685257, 10.7267494
19: -5.2832417, 4.7406106, -5.3030801, 4.7422628, -7.5608788, 7.5811386
20: -3.3974214, 7.9778309, -3.4074030, 7.9712300, -10.2653618, 10.2551956
21: -1.9377770, 8.8843307, -1.9502444, 8.8807507, -8.9205170, 8.9219284
22: -9.1974087, 2.7653849, -9.2220812, 2.7845235, -8.6703796, 8.6758938
23: 1.3576366, 12.5057678, 1.3554796, 12.5090446, -7.7262955, 7.7248058
24: -2.6847253, 10.4954910, -2.6845245, 10.4994602, -8.1643524, 8.1563911
25: 0.3467467, 13.7570515, 0.3810179, 13.7656832, -9.4085655, 9.3559875
26: -17.3801308, 2.4825957, -17.4252186, 2.4847610, -14.5570679, 14.5955048
27: -10.2644434, 6.2960753, -10.3239079, 6.2909422, -9.0993519, 9.1671181
28: 1.0912035, 13.5810261, 1.0766282, 13.5815315, -9.6223373, 9.6373863
29: -5.0826616, 8.3509283, -5.1180868, 8.3591938, -8.5982094, 8.6281281
30: 5.9789047, 17.7111149, 5.9861331, 17.7161942, -7.6703186, 7.6501675
31: -3.3735323, 10.3923092, -3.3888431, 10.3943405, -9.1542130, 9.1617279
32: -19.5883732, -2.7393074, -19.5980148, -2.7422297, -10.7047577, 10.7267189
33: -47.0235329, -21.5883217, -47.0297852, -21.5132542, -14.5748138, 14.4973450
34: -29.7135696, -10.5955582, -29.7229958, -10.5708714, -10.6490936, 10.6743584
35: -29.2106133, -9.9748735, -29.2195244, -9.9389839, -10.6877785, 10.6967049
36: -31.8772697, -9.4160089, -31.8920822, -9.3970175, -12.6936874, 12.7180557
37: -46.1224632, -23.5034180, -46.1285629, -23.4782372, -16.0555344, 16.0565872
38: -34.3016510, -11.5283279, -34.3087234, -11.5164375, -15.1118164, 15.1321945
39: -56.3113403, -30.6982613, -56.3138275, -30.6310692, -13.2212791, 13.1460075
40: -40.2418137, -23.3023186, -40.2492104, -23.2925320, -8.1922226, 8.1791897
41: -26.7355213, -6.9750929, -26.7477722, -6.9875021, -11.2402630, 11.2779160
42: -14.5328674, -2.1237779, -14.5382805, -2.0969224, -8.5264664, 8.5007534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0827451, upper bound: 5.0863947
time: 19.85 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0916038, upper bound: 5.0863947
time: 21.15 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.2292671, 0.7126267, -20.2138653, 0.6802926, -13.4310417, 13.4178276
1: -6.4341750, 5.2889500, -6.4203157, 5.2764692, -6.4095993, 6.3913059
2: -11.0234928, 2.2511849, -11.0135994, 2.2075958, -8.5362740, 8.5784340
3: -12.3404598, 3.4386511, -12.3261623, 3.3735371, -11.1249771, 11.1687775
4: -22.0456295, -5.6190543, -22.0349121, -5.6598148, -9.2172394, 9.2669449
5: -10.8015394, 5.6823034, -10.7929077, 5.6231904, -12.1400986, 12.1872025
6: -22.4961586, -4.4491353, -22.4805470, -4.5185995, -11.1482048, 11.1977768
7: -9.5641956, 8.9340019, -9.5527153, 8.9097738, -12.4524612, 12.4757500
8: -26.3553314, -5.5983834, -26.3307838, -5.6157122, -9.8275490, 9.7944527
9: -14.5897627, 2.1970990, -14.5759296, 2.1207042, -12.8847275, 12.9689636
10: -5.9061518, 11.7619705, -5.8914928, 11.7023325, -13.2034302, 13.2563248
11: 9.5411100, 21.1120014, 9.6153183, 21.1083698, -7.5357971, 7.4411602
12: -15.1712589, 9.8588152, -15.1320686, 9.8324347, -18.4816055, 18.4297409
13: -28.0418911, -3.0044141, -28.0246944, -3.0867832, -12.8609848, 12.9452820
14: -31.4311428, 0.6611576, -31.3258095, 0.6543527, -21.5626373, 21.4453278
15: -24.9342155, -10.6793280, -24.9259262, -10.6970329, -8.8563042, 8.8898697
16: -6.9504461, 7.9140606, -6.9433393, 7.8737445, -10.1089363, 10.1295586
17: -14.8140011, 11.7529860, -14.7098980, 11.7384453, -21.7592621, 21.6521530
18: -0.9415295, 12.5758362, -0.8516929, 12.5654144, -10.7344055, 10.6322098
19: -5.3164806, 4.7436442, -5.2673368, 4.7383809, -7.5941353, 7.5444851
20: -3.4146714, 7.9839659, -3.3876929, 7.9601660, -10.2386856, 10.2754517
21: -1.9635282, 8.8846836, -1.9203248, 8.8750963, -8.9214897, 8.9103737
22: -9.2339067, 2.7859750, -9.1946583, 2.7838392, -8.6834755, 8.6678352
23: 1.3244791, 12.5104446, 1.3912685, 12.5080805, -7.7638454, 7.6858368
24: -2.7218015, 10.4995871, -2.6412625, 10.4949026, -8.2033310, 8.1105881
25: 0.3222294, 13.7661743, 0.4101980, 13.7610073, -9.4254074, 9.3329391
26: -17.4419613, 2.4858561, -17.3576813, 2.4744449, -14.6023102, 14.5303040
27: -10.3257113, 6.2967196, -10.2553091, 6.2897367, -9.1613579, 9.0955982
28: 1.0497391, 13.5820198, 1.1246543, 13.5779266, -9.6682396, 9.5877457
29: -5.1277876, 8.3598309, -5.0679998, 8.3525524, -8.6362896, 8.5832977
30: 5.9546800, 17.7170219, 6.0161052, 17.7131767, -7.6839123, 7.6374722
31: -3.4088702, 10.3972273, -3.3492167, 10.3898211, -9.1779747, 9.1296272
32: -19.6004200, -2.7050965, -19.5841713, -2.7842264, -10.6727295, 10.7495155
33: -47.0364685, -21.5049610, -47.0244675, -21.6008854, -14.4794731, 14.6004715
34: -29.7223816, -10.5642910, -29.7082691, -10.6078262, -10.6497269, 10.6742249
35: -29.2286339, -9.9337778, -29.2192307, -9.9771795, -10.6869316, 10.7080383
36: -31.8933449, -9.3878193, -31.8766098, -9.4220762, -12.7004204, 12.7047043
37: -46.1419029, -23.4790459, -46.1243782, -23.5006618, -16.0740814, 16.0307083
38: -34.3110962, -11.5101728, -34.2895050, -11.5281668, -15.1350479, 15.0915985
39: -56.3204994, -30.6284637, -56.3085594, -30.7042866, -13.1309242, 13.2386360
40: -40.2533035, -23.2800999, -40.2492447, -23.3158455, -8.1670761, 8.2021790
41: -26.7501926, -6.9567747, -26.7361202, -7.0099025, -11.2471962, 11.2661629
42: -14.5441895, -2.0932660, -14.5345745, -2.1305695, -8.5045223, 8.5132313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0786455, upper bound: 5.0816382
time: 93.73 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0875254, upper bound: 5.0916036
time: 21.07 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.2275677, 0.6957548, -20.2211266, 0.6944768, -13.4411774, 13.4004517
1: -6.4287724, 5.2846255, -6.4223228, 5.2822533, -6.4112549, 6.3816700
2: -11.0235538, 2.2375264, -11.0210075, 2.2192459, -8.5543289, 8.5567417
3: -12.3288612, 3.4023387, -12.3301601, 3.3900743, -11.1364288, 11.1312141
4: -22.0506706, -5.6300936, -22.0474701, -5.6464081, -9.2404518, 9.2516670
5: -10.7984438, 5.6597710, -10.7980146, 5.6368933, -12.1536560, 12.1617966
6: -22.4830780, -4.4706278, -22.4847507, -4.5177102, -11.1385498, 11.1919632
7: -9.5633392, 8.9241219, -9.5613966, 8.9213905, -12.4707947, 12.4471321
8: -26.3491802, -5.6020379, -26.3438416, -5.6037016, -9.8340836, 9.7876091
9: -14.5676765, 2.1402254, -14.5736542, 2.1344192, -12.8901215, 12.8990250
10: -5.8999147, 11.7216568, -5.8943334, 11.7134037, -13.2101212, 13.2108650
11: 9.5698681, 21.1031227, 9.6006947, 21.1083260, -7.4954777, 7.4489517
12: -15.1594934, 9.8648310, -15.1580734, 9.8539228, -18.4487228, 18.4773636
13: -28.0153961, -3.0799704, -28.0160141, -3.0841172, -12.8551674, 12.8515472
14: -31.3780060, 0.6416016, -31.3484421, 0.6509581, -21.4972763, 21.4550934
15: -24.9291935, -10.6916809, -24.9219341, -10.6964378, -8.8708572, 8.8744659
16: -6.9429293, 7.8892426, -6.9437881, 7.8832731, -10.1127625, 10.1018314
17: -14.7581692, 11.7362347, -14.7345133, 11.7463856, -21.6965942, 21.6619720
18: -0.8788607, 12.5515804, -0.8692582, 12.5590477, -10.6577377, 10.6462593
19: -5.2851977, 4.7339921, -5.2718320, 4.7380371, -7.5650730, 7.5428391
20: -3.4022980, 7.9789505, -3.3948491, 7.9671736, -10.2470093, 10.2553825
21: -1.9397993, 8.8809547, -1.9261031, 8.8784418, -8.9114685, 8.8980255
22: -9.2078056, 2.7818151, -9.2016478, 2.7820306, -8.6620636, 8.6707134
23: 1.3524811, 12.5018749, 1.3834326, 12.5066757, -7.7305298, 7.6885910
24: -2.6848962, 10.4815960, -2.6472921, 10.4885759, -8.1582699, 8.1061497
25: 0.3436778, 13.7597609, 0.4013588, 13.7617836, -9.4038773, 9.3368874
26: -17.3894329, 2.4742818, -17.3775005, 2.4744501, -14.5472641, 14.5490723
27: -10.2621489, 6.2797384, -10.2605934, 6.2822628, -9.0889206, 9.0890865
28: 1.0892868, 13.5751972, 1.1157980, 13.5778141, -9.6287842, 9.5921402
29: -5.0867996, 8.3515472, -5.0798278, 8.3530159, -8.5875854, 8.5887375
30: 5.9729900, 17.7094193, 6.0023770, 17.7103786, -7.6676750, 7.6316166
31: -3.3768327, 10.3880329, -3.3550100, 10.3918495, -9.1527863, 9.1231079
32: -19.5744820, -2.7403049, -19.5801353, -2.7779245, -10.6599770, 10.7116127
33: -47.0172615, -21.5708790, -47.0242653, -21.5789089, -14.4894447, 14.5099258
34: -29.7123222, -10.5891275, -29.7163887, -10.5960026, -10.6348038, 10.6451302
35: -29.2132568, -9.9627314, -29.2148228, -9.9680405, -10.6617317, 10.6906548
36: -31.8754978, -9.4135094, -31.8804226, -9.4225903, -12.6636543, 12.6958275
37: -46.1344528, -23.4943085, -46.1305237, -23.4954605, -16.0405045, 16.0589218
38: -34.2987862, -11.5238609, -34.2963715, -11.5272713, -15.1023178, 15.1064072
39: -56.3068314, -30.6870327, -56.3123703, -30.6892853, -13.1383934, 13.1529160
40: -40.2426529, -23.2975292, -40.2472725, -23.3105507, -8.1581879, 8.1934490
41: -26.7389297, -6.9731069, -26.7397346, -7.0044947, -11.2244053, 11.2639885
42: -14.5257483, -2.1218224, -14.5272865, -2.1261659, -8.4890289, 8.5038357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0827451, upper bound: 5.0841035
time: 17.33 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0916038, upper bound: 5.0841035
time: 18.31 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2338505, 0.7131882, -20.2230396, 0.6953647, -13.4501762, 13.4200478
1: -6.4353075, 5.2891202, -6.4228973, 5.2823887, -6.4165688, 6.3919659
2: -11.0278282, 2.2516921, -11.0213566, 2.2202601, -8.5525322, 8.5837860
3: -12.3480511, 3.4393787, -12.3401585, 3.3910630, -11.1498451, 11.1775398
4: -22.0525818, -5.6184750, -22.0477085, -5.6454196, -9.2389336, 9.2757301
5: -10.8062954, 5.6829643, -10.8018494, 5.6376724, -12.1600189, 12.1969376
6: -22.4971962, -4.4500766, -22.4888306, -4.5174828, -11.1536369, 11.2157593
7: -9.5690260, 8.9344254, -9.5630836, 8.9221182, -12.4690781, 12.4854050
8: -26.3629684, -5.5982628, -26.3447418, -5.6028104, -9.8477325, 9.7996521
9: -14.5955639, 2.1983817, -14.5876503, 2.1366422, -12.9066620, 12.9747238
10: -5.9069948, 11.7636137, -5.8960490, 11.7157898, -13.2208672, 13.2608185
11: 9.5401421, 21.1154480, 9.5995770, 21.1148129, -7.5341234, 7.4566269
12: -15.1722040, 9.8717155, -15.1596928, 9.8556366, -18.5030518, 18.4703522
13: -28.0422249, -3.0023041, -28.0300751, -3.0815041, -12.8716507, 12.9515839
14: -31.4340439, 0.6650109, -31.3515167, 0.6620510, -21.5633392, 21.4769516
15: -24.9316769, -10.6784697, -24.9229202, -10.6952744, -8.8763046, 8.8899059
16: -6.9515724, 7.9143696, -6.9469662, 7.8839183, -10.1206665, 10.1303329
17: -14.8148241, 11.7659464, -14.7363605, 11.7625198, -21.7767334, 21.6911545
18: -0.9433947, 12.5804758, -0.8714094, 12.5746422, -10.7392349, 10.6606445
19: -5.3174868, 4.7437658, -5.2728262, 4.7430997, -7.6014099, 7.5492401
20: -3.4166310, 7.9843812, -3.3963363, 7.9678335, -10.2512856, 10.2841530
21: -1.9649980, 8.8848648, -1.9273384, 8.8801918, -8.9322929, 8.9168510
22: -9.2343464, 2.7866662, -9.2023506, 2.7838602, -8.6901188, 8.6764011
23: 1.3235753, 12.5105057, 1.3827078, 12.5105362, -7.7666912, 7.6941833
24: -2.7224400, 10.4999657, -2.6477227, 10.4970608, -8.2065964, 8.1177025
25: 0.3212829, 13.7665243, 0.4004545, 13.7642002, -9.4292526, 9.3433380
26: -17.4432144, 2.4907503, -17.3789330, 2.4833958, -14.6084709, 14.5589218
27: -10.3273735, 6.2968307, -10.2629900, 6.2914186, -9.1645756, 9.1079121
28: 1.0485492, 13.5820580, 1.1146874, 13.5812225, -9.6728363, 9.5979233
29: -5.1282301, 8.3621206, -5.0809212, 8.3579826, -8.6372204, 8.5976334
30: 5.9540586, 17.7169266, 6.0016594, 17.7138958, -7.6864243, 7.6424141
31: -3.4107687, 10.3973732, -3.3563018, 10.3968830, -9.1903191, 9.1366825
32: -19.6015587, -2.7042315, -19.5900860, -2.7768800, -10.6808052, 10.7567978
33: -47.0426636, -21.5033569, -47.0370560, -21.5770226, -14.5065193, 14.5962067
34: -29.7288780, -10.5637112, -29.7216873, -10.5955601, -10.6671028, 10.6712379
35: -29.2281227, -9.9331169, -29.2204628, -9.9673672, -10.6921310, 10.7116508
36: -31.8942890, -9.3872728, -31.8847218, -9.4216213, -12.7002068, 12.7191582
37: -46.1432877, -23.4765472, -46.1330338, -23.4937687, -16.0793762, 16.0612869
38: -34.3129616, -11.5098095, -34.2996140, -11.5264645, -15.1365204, 15.1151886
39: -56.3279114, -30.6276207, -56.3224640, -30.6882992, -13.1516151, 13.2342987
40: -40.2547226, -23.2798386, -40.2527657, -23.3103371, -8.1667252, 8.2179794
41: -26.7511501, -6.9562273, -26.7415142, -7.0040760, -11.2490673, 11.2812004
42: -14.5449867, -2.0925322, -14.5349340, -2.1257634, -8.5047054, 8.5337734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=89, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 593

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0827451, upper bound: 5.0916036
time: 18.63 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0916038, upper bound: 5.0916036
time: 24.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 45.03 seconds
IS_A1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0827143, upper bound: 5.0724041
IS_A1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0752063, upper bound: 5.0724041
IS_A1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0827143, upper bound: 5.0764985
IS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0915595, upper bound: 5.0764985
IS_A2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0827451, upper bound: 5.0822870
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0916038, upper bound: 5.0822870
IS_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0827451, upper bound: 5.0863947
IS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0916038, upper bound: 5.0863947
IS_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0786455, upper bound: 5.0816382
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0875254, upper bound: 5.0916036
IS_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0827451, upper bound: 5.0841035
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0916038, upper bound: 5.0841035
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0827451, upper bound: 5.0916036
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 45.03
Output dim: 11, lower bound: -5.0916038, upper bound: 5.0916036

## BFS IS instance: IS_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -20.2110519, 0.6732404, -20.2444477, 0.7051272, -13.3985977, 13.4173431
1: -6.4170017, 5.2721729, -6.4481630, 5.2843275, -6.3791924, 6.4019852
2: -11.0038815, 2.2044699, -11.0214624, 2.2330644, -8.5570297, 8.5296497
3: -12.3273640, 3.3798733, -12.3431358, 3.4248292, -11.1554375, 11.1312370
4: -22.0363407, -5.6556768, -22.0520439, -5.6387897, -9.2451820, 9.2357445
5: -10.7805071, 5.6181269, -10.7996521, 5.6633978, -12.1687546, 12.1330452
6: -22.4573803, -4.5178661, -22.4813671, -4.4536152, -11.1777000, 11.1313324
7: -9.5437851, 8.9004307, -9.5697823, 8.9226189, -12.4691010, 12.4455490
8: -26.3315201, -5.6156244, -26.3776398, -5.6068077, -9.7889366, 9.8375015
9: -14.5795021, 2.1269436, -14.6025324, 2.1893647, -12.9533539, 12.9101715
10: -5.8894787, 11.7061663, -5.9167027, 11.7517405, -13.2362862, 13.2253723
11: 9.6253080, 21.0920601, 9.5575876, 21.1041183, -7.4269657, 7.4944763
12: -15.1379786, 9.8464670, -15.1643333, 9.8798523, -18.4486160, 18.4890518
13: -28.0259781, -3.0913789, -28.0301304, -3.0005867, -12.9406738, 12.8437042
14: -31.3391018, 0.6362829, -31.4290504, 0.6551642, -21.4474258, 21.5214844
15: -24.9079380, -10.7221546, -24.9328308, -10.6881371, -8.8649139, 8.8734665
16: -6.9402828, 7.8818741, -6.9578619, 7.9098768, -10.1132202, 10.1107693
17: -14.7266941, 11.7396631, -14.7968369, 11.7556839, -21.6822968, 21.7307510
18: -0.8699410, 12.5691242, -0.9372094, 12.5788345, -10.6589775, 10.7227440
19: -5.2675223, 4.7373590, -5.3046122, 4.7409420, -7.5406036, 7.5783958
20: -3.3856437, 7.9642639, -3.4080801, 7.9793739, -10.2597542, 10.2343407
21: -1.9212453, 8.8774967, -1.9551802, 8.8801365, -8.9031067, 8.9179230
22: -9.1890841, 2.7612972, -9.2238331, 2.7836542, -8.6590958, 8.6694298
23: 1.3906085, 12.4933004, 1.3441646, 12.5026865, -7.6813564, 7.7272015
24: -2.6459134, 10.4784346, -2.7087030, 10.4917431, -8.1072769, 8.1688652
25: 0.4073086, 13.7289152, 0.3499198, 13.7511883, -9.3242531, 9.3664284
26: -17.3632603, 2.4742811, -17.4250069, 2.4835627, -14.5370598, 14.5890121
27: -10.2574959, 6.2874451, -10.3233223, 6.2900524, -9.0972900, 9.1551456
28: 1.1206770, 13.5714703, 1.0722110, 13.5768042, -9.5839615, 9.6344185
29: -5.0735540, 8.3444881, -5.1200328, 8.3570662, -8.5840111, 8.6209068
30: 6.0110197, 17.6950436, 5.9603024, 17.7089195, -7.6209450, 7.6639023
31: -3.3489799, 10.3830509, -3.3891256, 10.3875332, -9.1245575, 9.1546574
32: -19.5624466, -2.7781246, -19.5875149, -2.6999998, -10.7250671, 10.6585617
33: -47.0140457, -21.5994720, -47.0283318, -21.4996910, -14.5730820, 14.4776955
34: -29.7041264, -10.6036415, -29.7216358, -10.5593529, -10.6525497, 10.6561241
35: -29.2030487, -9.9817104, -29.2182426, -9.9193659, -10.7013779, 10.6764297
36: -31.8723679, -9.4269981, -31.8923683, -9.3606997, -12.7202911, 12.6818275
37: -46.1109657, -23.5080490, -46.1236267, -23.4771729, -16.0304947, 16.0444336
38: -34.2917252, -11.5349426, -34.3085327, -11.4829607, -15.1353073, 15.1098175
39: -56.3025856, -30.7031002, -56.3128319, -30.6147690, -13.2237968, 13.1247559
40: -40.2288818, -23.3155441, -40.2442627, -23.2823143, -8.1921959, 8.1586189
41: -26.7157097, -7.0075483, -26.7394733, -6.9598784, -11.2511368, 11.2284088
42: -14.5238752, -2.1291263, -14.5379372, -2.0916810, -8.5226460, 8.4924793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1378

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1681

## Relational analysis of IS_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0910712, upper bound: 5.0695863
time: 20.42 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0912781, upper bound: 5.0762216
time: 16.20 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -20.2104416, 0.6634147, -20.2402859, 0.7026098, -13.4239044, 13.4131165
1: -6.4214458, 5.2712455, -6.4472399, 5.2829380, -6.3989296, 6.4054184
2: -11.0068016, 2.2135210, -11.0215378, 2.2328017, -8.5597382, 8.5402222
3: -12.3222189, 3.3771300, -12.3377457, 3.4242961, -11.1553268, 11.1250381
4: -22.0329647, -5.6510363, -22.0490685, -5.6389761, -9.2429962, 9.2377510
5: -10.7841139, 5.6304297, -10.8003359, 5.6630979, -12.1691055, 12.1449127
6: -22.4755344, -4.4745469, -22.4927769, -4.4526110, -11.1804504, 11.1830292
7: -9.5387964, 8.8963318, -9.5663452, 8.9232368, -12.4659500, 12.4366951
8: -26.3245850, -5.6226273, -26.3713531, -5.6066036, -9.7972298, 9.8201981
9: -14.5743294, 2.1191709, -14.5971241, 2.1883874, -12.9576187, 12.8972473
10: -5.8941021, 11.7040653, -5.9164543, 11.7509050, -13.2433701, 13.2195015
11: 9.6068478, 21.1030369, 9.5573444, 21.1089783, -7.4555159, 7.5099754
12: -15.1206236, 9.8316784, -15.1648636, 9.8674488, -18.4209213, 18.4822464
13: -28.0349617, -3.0841746, -28.0304451, -2.9996076, -12.9531097, 12.8458862
14: -31.3460941, 0.6385889, -31.4270287, 0.6546295, -21.4630356, 21.5368881
15: -24.9160233, -10.7158527, -24.9356041, -10.6882811, -8.8739166, 8.8626804
16: -6.9402428, 7.8781052, -6.9573517, 7.9089227, -10.1274414, 10.1080322
17: -14.7255030, 11.7232857, -14.7968273, 11.7458897, -21.6757507, 21.7249680
18: -0.8620028, 12.5665112, -0.9357407, 12.5749140, -10.6454697, 10.7257957
19: -5.2773600, 4.7358356, -5.3042870, 4.7415676, -7.5550022, 7.5756035
20: -3.3875167, 7.9700184, -3.4060764, 7.9791732, -10.2634430, 10.2402763
21: -1.9304256, 8.8789482, -1.9547812, 8.8809261, -8.9127731, 8.9152012
22: -9.1893406, 2.7650893, -9.2239590, 2.7840698, -8.6602898, 8.6695633
23: 1.3665763, 12.5030308, 1.3446336, 12.5087042, -7.7128029, 7.7335300
24: -2.6780109, 10.4925613, -2.7084169, 10.4989109, -8.1513214, 8.1776733
25: 0.3566959, 13.7529469, 0.3502150, 13.7644539, -9.3892727, 9.3830070
26: -17.3584251, 2.4719536, -17.4247208, 2.4789412, -14.5277138, 14.5902100
27: -10.2535458, 6.2941704, -10.3211746, 6.2906895, -9.0915642, 9.1646557
28: 1.1017153, 13.5775108, 1.0727093, 13.5812426, -9.6082344, 9.6376801
29: -5.0694733, 8.3442011, -5.1200671, 8.3559418, -8.5806675, 8.6256485
30: 5.9935846, 17.7095547, 5.9600940, 17.7159843, -7.6562271, 7.6736870
31: -3.3655102, 10.3850765, -3.3880780, 10.3910246, -9.1448593, 9.1509705
32: -19.5809822, -2.7468219, -19.5974827, -2.7005641, -10.7380981, 10.7061272
33: -47.0104294, -21.6125984, -47.0239716, -21.5005188, -14.5890236, 14.4633789
34: -29.6993790, -10.6080532, -29.7183228, -10.5594807, -10.6633186, 10.6545258
35: -29.2085514, -9.9847851, -29.2205429, -9.9196205, -10.7044868, 10.6841660
36: -31.8678303, -9.4165754, -31.8913689, -9.3606977, -12.7147675, 12.7043114
37: -46.1134682, -23.5125866, -46.1234474, -23.4811001, -16.0248108, 16.0487671
38: -34.2902451, -11.5301609, -34.3085175, -11.4830532, -15.1212463, 15.1194458
39: -56.2966614, -30.7147312, -56.3067245, -30.6154823, -13.2361755, 13.1106148
40: -40.2378845, -23.3078918, -40.2481689, -23.2825222, -8.1859550, 8.1773586
41: -26.7288761, -6.9811234, -26.7472420, -6.9599924, -11.2517929, 11.2677155
42: -14.5320549, -2.1288941, -14.5385523, -2.0921545, -8.5112991, 8.4990005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1681

## Relational analysis of IS_A2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0846268, upper bound: 5.0818151
time: 15.01 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0913255, upper bound: 5.0820123
time: 22.69 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -20.2195778, 0.6785460, -20.2448845, 0.7031074, -13.4261169, 13.4322433
1: -6.4240232, 5.2771940, -6.4483376, 5.2830958, -6.3995667, 6.4123936
2: -11.0145502, 2.2261665, -11.0258799, 2.2333107, -8.5650864, 8.5564842
3: -12.3362379, 3.3947148, -12.3453608, 3.4250433, -11.1640739, 11.1498909
4: -22.0457993, -5.6366348, -22.0560074, -5.6383805, -9.2517776, 9.2594414
5: -10.7930222, 5.6449041, -10.8051052, 5.6637859, -12.1788406, 12.1648254
6: -22.4838295, -4.4734249, -22.4938107, -4.4535885, -11.1984215, 11.1884346
7: -9.5491962, 8.9086504, -9.5711794, 8.9237518, -12.4756165, 12.4533272
8: -26.3386002, -5.6097298, -26.3790703, -5.6064396, -9.8024139, 9.8403854
9: -14.5860300, 2.1350811, -14.6029549, 2.1896908, -12.9633865, 12.9192123
10: -5.8986177, 11.7175560, -5.9172897, 11.7525272, -13.2478790, 13.2369461
11: 9.5911045, 21.1094570, 9.5563469, 21.1124287, -7.4709749, 7.5082855
12: -15.1482525, 9.8549604, -15.1658535, 9.8803215, -18.4615021, 18.5036621
13: -28.0402870, -3.0789526, -28.0307579, -2.9975960, -12.9594116, 12.8565750
14: -31.3718681, 0.6462808, -31.4299240, 0.6584189, -21.4946518, 21.5376129
15: -24.9130192, -10.7141037, -24.9330769, -10.6874342, -8.8739395, 8.8826733
16: -6.9438801, 7.8883090, -6.9584956, 7.9092112, -10.1281967, 10.1197281
17: -14.7520323, 11.7473450, -14.7976065, 11.7588320, -21.7147522, 21.7424545
18: -0.8817582, 12.5757475, -0.9375916, 12.5795679, -10.6739235, 10.7306328
19: -5.2828360, 4.7405586, -5.3052726, 4.7417068, -7.5597382, 7.5828781
20: -3.3961594, 7.9776707, -3.4080555, 7.9795985, -10.2721558, 10.2528801
21: -1.9374282, 8.8840809, -1.9562372, 8.8811207, -8.9192429, 8.9259930
22: -9.1970549, 2.7650940, -9.2244148, 2.7847342, -8.6688652, 8.6762218
23: 1.3580056, 12.5055056, 1.3436980, 12.5087328, -7.7211418, 7.7363815
24: -2.6844697, 10.4947205, -2.7090793, 10.4992828, -8.1584549, 8.1809464
25: 0.3469453, 13.7561378, 0.3493176, 13.7648048, -9.3996773, 9.3868370
26: -17.3796349, 2.4808974, -17.4259739, 2.4838195, -14.5563354, 14.5963669
27: -10.2612352, 6.2958536, -10.3228540, 6.2907691, -9.1038780, 9.1678658
28: 1.0917432, 13.5808125, 1.0715108, 13.5813341, -9.6184120, 9.6422653
29: -5.0824227, 8.3496456, -5.1205416, 8.3582087, -8.5949955, 8.6265717
30: 5.9791346, 17.7102737, 5.9594417, 17.7158928, -7.6611900, 7.6762085
31: -3.3726118, 10.3921232, -3.3899851, 10.3911781, -9.1519203, 9.1633224
32: -19.5869293, -2.7394588, -19.5985870, -2.6996796, -10.7453880, 10.7141895
33: -47.0230408, -21.5886917, -47.0301666, -21.4989376, -14.5847702, 14.4904327
34: -29.7128448, -10.5957785, -29.7248306, -10.5588779, -10.6603546, 10.6718979
35: -29.2098312, -9.9750071, -29.2200317, -9.9189320, -10.7081299, 10.6894341
36: -31.8758640, -9.4161520, -31.8922844, -9.3601074, -12.7292480, 12.7040825
37: -46.1220856, -23.5057087, -46.1248322, -23.4786110, -16.0553741, 16.0540848
38: -34.3004074, -11.5284939, -34.3104095, -11.4827299, -15.1448746, 15.1208954
39: -56.3106346, -30.6986961, -56.3142052, -30.6146564, -13.2318344, 13.1312752
40: -40.2413712, -23.3023987, -40.2495766, -23.2821980, -8.2017593, 8.1769962
41: -26.7342529, -6.9753084, -26.7482109, -6.9594226, -11.2668724, 11.2695923
42: -14.5324402, -2.1241078, -14.5393362, -2.0914373, -8.5318375, 8.4991856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1681

## Relational analysis of IS_A2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0846268, upper bound: 5.0859179
time: 25.93 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0913255, upper bound: 5.0861191
time: 20.36 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -20.2287827, 0.7119608, -20.2323914, 0.6811845, -13.4236526, 13.4376335
1: -6.4341354, 5.2883677, -6.4401126, 5.2771406, -6.4044647, 6.4103832
2: -11.0229244, 2.2510433, -11.0137348, 2.2154131, -8.5413780, 8.5752831
3: -12.3391857, 3.4385142, -12.3274107, 3.3769727, -11.1252899, 11.1698112
4: -22.0454597, -5.6195841, -22.0397110, -5.6593294, -9.2132225, 9.2732964
5: -10.8008118, 5.6821299, -10.7935123, 5.6362519, -12.1507034, 12.1839600
6: -22.4947586, -4.4492254, -22.4800930, -4.4781127, -11.1865349, 11.1852798
7: -9.5636988, 8.9337358, -9.5580254, 8.9114037, -12.4521408, 12.4808006
8: -26.3551521, -5.5989103, -26.3513775, -5.6154165, -9.8218842, 9.8141251
9: -14.5895987, 2.1945832, -14.5897923, 2.1194198, -12.8794632, 12.9859467
10: -5.9059854, 11.7591667, -5.9107513, 11.7014923, -13.1959114, 13.2785187
11: 9.5414810, 21.1114120, 9.5980339, 21.1083145, -7.5295105, 7.4585037
12: -15.1707640, 9.8580055, -15.1347427, 9.8485336, -18.4975662, 18.4280548
13: -28.0407104, -3.0048406, -28.0241852, -3.0752575, -12.8626862, 12.9442101
14: -31.4305267, 0.6602571, -31.3523865, 0.6550632, -21.5550690, 21.4729385
15: -24.9340744, -10.6816349, -24.9394379, -10.6992397, -8.8505020, 8.9068584
16: -6.9500360, 7.9134512, -6.9530220, 7.8745055, -10.1054344, 10.1391697
17: -14.8137484, 11.7517815, -14.7161140, 11.7377443, -21.7596741, 21.6547623
18: -0.9408915, 12.5756178, -0.8567920, 12.5679474, -10.7398033, 10.6361046
19: -5.3160906, 4.7435846, -5.2695346, 4.7378263, -7.5930061, 7.5462513
20: -3.4133835, 7.9838004, -3.3883426, 7.9685431, -10.2454834, 10.2731400
21: -1.9631833, 8.8844357, -1.9263339, 8.8754578, -8.9202118, 8.9144554
22: -9.2335281, 2.7857101, -9.1969967, 2.7840528, -8.6819458, 8.6681557
23: 1.3248677, 12.5101738, 1.3795199, 12.5077763, -7.7586765, 7.6974087
24: -2.7215230, 10.4988184, -2.6658192, 10.4947376, -8.1974297, 8.1351566
25: 0.3224177, 13.7652407, 0.3785126, 13.7601519, -9.4165192, 9.3637848
26: -17.4414978, 2.4841671, -17.3584633, 2.4735186, -14.6015434, 14.5311737
27: -10.3224812, 6.2964973, -10.2543144, 6.2895479, -9.1659050, 9.0963440
28: 1.0502553, 13.5817986, 1.1195269, 13.5777435, -9.6643143, 9.5926552
29: -5.1275444, 8.3585701, -5.0704880, 8.3515606, -8.6330662, 8.5817719
30: 5.9549484, 17.7161999, 5.9894180, 17.7128353, -7.6748009, 7.6635170
31: -3.4079635, 10.3970346, -3.3503618, 10.3866558, -9.1756935, 9.1312160
32: -19.5990009, -2.7052772, -19.5847321, -2.7416942, -10.7133636, 10.7370300
33: -47.0359459, -21.5053444, -47.0248718, -21.5865822, -14.4894562, 14.5935745
34: -29.7216396, -10.5644703, -29.7100868, -10.5958538, -10.6609802, 10.6717949
35: -29.2278557, -9.9339123, -29.2196751, -9.9570885, -10.7072716, 10.7007904
36: -31.8920441, -9.3879757, -31.8768272, -9.3852062, -12.7359962, 12.6907539
37: -46.1415138, -23.4813156, -46.1206551, -23.5010414, -16.0739365, 16.0281830
38: -34.3098030, -11.5102911, -34.2911453, -11.4944620, -15.1681290, 15.0802956
39: -56.3197937, -30.6289101, -56.3088684, -30.6877708, -13.1415367, 13.2239647
40: -40.2528114, -23.2801971, -40.2496567, -23.3055115, -8.1766167, 8.1999893
41: -26.7489147, -6.9570189, -26.7365189, -6.9818382, -11.2738342, 11.2578430
42: -14.5437717, -2.0935788, -14.5356197, -2.1250730, -8.5098953, 8.5116768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1681

## Relational analysis of IS_A2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0806020, upper bound: 5.0911179
time: 21.68 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0872483, upper bound: 5.0913254
time: 21.70 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.2271099, 0.6950755, -20.2396469, 0.6953082, -13.4337883, 13.4202271
1: -6.4287024, 5.2840471, -6.4421129, 5.2829847, -6.4061050, 6.4007473
2: -11.0229845, 2.2373855, -11.0211153, 2.2271023, -8.5594444, 8.5535851
3: -12.3275566, 3.4021840, -12.3314457, 3.3935084, -11.1366959, 11.1322746
4: -22.0504856, -5.6306410, -22.0522747, -5.6458564, -9.2364578, 9.2580147
5: -10.7977161, 5.6595860, -10.7985802, 5.6499405, -12.1642609, 12.1585693
6: -22.4817047, -4.4707479, -22.4842339, -4.4771938, -11.1768799, 11.1794624
7: -9.5628052, 8.9238415, -9.5667238, 8.9230061, -12.4704590, 12.4521866
8: -26.3490276, -5.6026082, -26.3643951, -5.6034150, -9.8284149, 9.8072853
9: -14.5675421, 2.1377072, -14.5875654, 2.1331227, -12.8848572, 12.9160385
10: -5.8997350, 11.7188835, -5.9136586, 11.7125168, -13.2025719, 13.2330437
11: 9.5702438, 21.1025352, 9.5834026, 21.1082935, -7.4891748, 7.4663010
12: -15.1590252, 9.8640156, -15.1607971, 9.8700914, -18.4646606, 18.4756622
13: -28.0142326, -3.0803986, -28.0155239, -3.0726042, -12.8568649, 12.8504639
14: -31.3774166, 0.6406627, -31.3750210, 0.6516511, -21.4897537, 21.4827271
15: -24.9290676, -10.6940174, -24.9354267, -10.6986675, -8.8650742, 8.8914471
16: -6.9425263, 7.8886280, -6.9534779, 7.8840146, -10.1092339, 10.1114330
17: -14.7578936, 11.7349672, -14.7407761, 11.7456884, -21.6969986, 21.6646652
18: -0.8782010, 12.5513229, -0.8743827, 12.5615835, -10.6631241, 10.6501503
19: -5.2848005, 4.7339091, -5.2740335, 4.7374911, -7.5639343, 7.5446091
20: -3.4010196, 7.9787860, -3.3954916, 7.9755306, -10.2538261, 10.2530479
21: -1.9394512, 8.8807049, -1.9320941, 8.8787966, -8.9101982, 8.9021053
22: -9.2074442, 2.7815452, -9.2039785, 2.7822762, -8.6605587, 8.6710415
23: 1.3528537, 12.5016193, 1.3716615, 12.5063877, -7.7253780, 7.7001514
24: -2.6846502, 10.4808369, -2.6718411, 10.4884062, -8.1523762, 8.1306953
25: 0.3438950, 13.7588310, 0.3696752, 13.7609186, -9.3950024, 9.3677483
26: -17.3889751, 2.4725838, -17.3782711, 2.4735155, -14.5465088, 14.5499420
27: -10.2589331, 6.2795396, -10.2595739, 6.2821131, -9.0934486, 9.0898304
28: 1.0898032, 13.5749836, 1.1107047, 13.5776014, -9.6248550, 9.5970421
29: -5.0865474, 8.3502808, -5.0822840, 8.3520308, -8.5843735, 8.5872192
30: 5.9732428, 17.7086067, 5.9757051, 17.7100563, -7.6585598, 7.6576672
31: -3.3758972, 10.3878546, -3.3561633, 10.3886890, -9.1504898, 9.1247101
32: -19.5730515, -2.7404613, -19.5806770, -2.7353811, -10.7005882, 10.6990967
33: -47.0168152, -21.5712318, -47.0247116, -21.5645676, -14.4994240, 14.5030060
34: -29.7115936, -10.5893688, -29.7181854, -10.5840015, -10.6460686, 10.6426773
35: -29.2124748, -9.9628334, -29.2153320, -9.9479952, -10.6821060, 10.6833916
36: -31.8741341, -9.4136353, -31.8806667, -9.3857222, -12.6992340, 12.6818619
37: -46.1340981, -23.4965935, -46.1268158, -23.4958572, -16.0403214, 16.0564117
38: -34.2975540, -11.5240555, -34.2980423, -11.4935188, -15.1353912, 15.0951233
39: -56.3061256, -30.6874523, -56.3127022, -30.6727581, -13.1489906, 13.1381836
40: -40.2422104, -23.2975941, -40.2476807, -23.3002300, -8.1677208, 8.1912498
41: -26.7377167, -6.9733725, -26.7401352, -6.9763803, -11.2510033, 11.2556725
42: -14.5253067, -2.1221180, -14.5283403, -2.1207035, -8.4943924, 8.5022678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1681

## Relational analysis of IS_A2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0846268, upper bound: 5.0836146
time: 21.99 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0913255, upper bound: 5.0838227
time: 20.42 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -20.2329617, 0.7033277, -20.2215843, 0.6792560, -13.4315948, 13.4080238
1: -6.4351182, 5.2789316, -6.4225926, 5.2653351, -6.3990211, 6.3812809
2: -11.0221577, 2.2499013, -11.0119333, 2.2173333, -8.5434036, 8.5721664
3: -12.3456411, 3.4388506, -12.3363380, 3.3902392, -11.1462250, 11.1730576
4: -22.0515022, -5.6209731, -22.0459404, -5.6496048, -9.2309151, 9.2694626
5: -10.8008947, 5.6819601, -10.7929420, 5.6360140, -12.1517487, 12.1861343
6: -22.4733791, -4.4505835, -22.4489899, -4.5182853, -11.1286011, 11.1757050
7: -9.5686579, 8.9310493, -9.5625038, 8.9164925, -12.4626617, 12.4811172
8: -26.3623466, -5.6080961, -26.3438263, -5.6191821, -9.8303337, 9.7886276
9: -14.5942030, 2.1898313, -14.5855322, 2.1222391, -12.8895111, 12.9630318
10: -5.9059925, 11.7504940, -5.8945217, 11.6940346, -13.1962509, 13.2448120
11: 9.5410633, 21.1054039, 9.6010666, 21.0980492, -7.5165720, 7.4449654
12: -15.1651287, 9.8708477, -15.1477966, 9.8542614, -18.4945374, 18.4578781
13: -28.0369320, -3.0045235, -28.0212250, -3.0850971, -12.8599014, 12.9402618
14: -31.4326534, 0.6519775, -31.3492508, 0.6406939, -21.5384674, 21.4602737
15: -24.9308777, -10.6866550, -24.9217415, -10.7087917, -8.8614464, 8.8800659
16: -6.9507236, 7.9097018, -6.9455395, 7.8761754, -10.1114883, 10.1238804
17: -14.8133936, 11.7638435, -14.7340107, 11.7590227, -21.7709885, 21.6855011
18: -0.9418032, 12.5791016, -0.8688173, 12.5724068, -10.7342186, 10.6545143
19: -5.3159671, 4.7431731, -5.2703705, 4.7421284, -7.5981979, 7.5453453
20: -3.4125197, 7.9833441, -3.3894866, 7.9660702, -10.2444992, 10.2754898
21: -1.9636254, 8.8822746, -1.9251225, 8.8758469, -8.9267006, 8.9120865
22: -9.2314672, 2.7857161, -9.1976585, 2.7823086, -8.6836929, 8.6688709
23: 1.3246214, 12.5056934, 1.3844254, 12.5025272, -7.7575436, 7.6875916
24: -2.7215953, 10.4865227, -2.6463375, 10.4746590, -8.1833992, 8.1030045
25: 0.3219428, 13.7498245, 0.4014969, 13.7362604, -9.4010620, 9.3258438
26: -17.4408817, 2.4901111, -17.3752518, 2.4823561, -14.6052856, 14.5546341
27: -10.3221350, 6.2954803, -10.2543354, 6.2891922, -9.1549187, 9.0930634
28: 1.0510707, 13.5785971, 1.1188176, 13.5754347, -9.6638222, 9.5894623
29: -5.1252475, 8.3587208, -5.0760584, 8.3523741, -8.6253834, 8.5857849
30: 5.9545617, 17.7024040, 6.0025220, 17.6895695, -7.6615143, 7.6268864
31: -3.4097764, 10.3957405, -3.3546891, 10.3941879, -9.1860771, 9.1328125
32: -19.5758858, -2.7047501, -19.5472374, -2.7777169, -10.6535568, 10.7127495
33: -47.0353737, -21.5052948, -47.0248680, -21.5801353, -14.4938698, 14.5804825
34: -29.7180843, -10.5640726, -29.7036991, -10.5961609, -10.6554604, 10.6525040
35: -29.2157326, -9.9335423, -29.1997147, -9.9681044, -10.6786423, 10.6902008
36: -31.8693466, -9.3876705, -31.8429871, -9.4222660, -12.6739464, 12.6765327
37: -46.1415520, -23.4772301, -46.1299820, -23.4948997, -16.0743179, 16.0559158
38: -34.2927322, -11.5100670, -34.2657623, -11.5268631, -15.1155777, 15.0813522
39: -56.3183289, -30.6286964, -56.3064423, -30.6900158, -13.1344948, 13.2105598
40: -40.2488403, -23.2799263, -40.2429619, -23.3105354, -8.1607056, 8.2083702
41: -26.7308979, -6.9566045, -26.7077274, -7.0048132, -11.2274323, 11.2462997
42: -14.5417767, -2.0931945, -14.5296211, -2.1267843, -8.4996624, 8.5267372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1681

## Relational analysis of IS_A2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0757610, upper bound: 5.0911179
time: 16.04 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0824649, upper bound: 5.0913254
time: 30.71 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20.2333736, 0.7125075, -20.2416039, 0.6962717, -13.4427910, 13.4398460
1: -6.4352360, 5.2885323, -6.4426770, 5.2830968, -6.4114189, 6.4110355
2: -11.0272341, 2.2515521, -11.0214710, 2.2281091, -8.5576439, 8.5806255
3: -12.3467598, 3.4392395, -12.3414268, 3.3945098, -11.1501389, 11.1785698
4: -22.0524101, -5.6189833, -22.0525055, -5.6449213, -9.2349319, 9.2820778
5: -10.8055668, 5.6827536, -10.8024168, 5.6507525, -12.1706238, 12.1936913
6: -22.4958019, -4.4502039, -22.4883652, -4.4769955, -11.1919556, 11.2032776
7: -9.5685396, 8.9341736, -9.5683956, 8.9237804, -12.4687653, 12.4904442
8: -26.3628006, -5.5987587, -26.3653069, -5.6024790, -9.8420677, 9.8193321
9: -14.5953617, 2.1958570, -14.6014996, 2.1353049, -12.9013672, 12.9917412
10: -5.9068451, 11.7608042, -5.9152942, 11.7149391, -13.2133408, 13.2829895
11: 9.5404758, 21.1148624, 9.5822973, 21.1147518, -7.5278378, 7.4739742
12: -15.1717339, 9.8708706, -15.1623755, 9.8718071, -18.5189362, 18.4686127
13: -28.0410252, -3.0028508, -28.0295258, -3.0699730, -12.8733749, 12.9505196
14: -31.4334373, 0.6641240, -31.3781033, 0.6627481, -21.5558167, 21.5045929
15: -24.9315529, -10.6807928, -24.9364357, -10.6975193, -8.8704948, 8.9068890
16: -6.9511871, 7.9137683, -6.9566708, 7.8846531, -10.1171417, 10.1399364
17: -14.8145275, 11.7646942, -14.7425842, 11.7618084, -21.7771683, 21.6938782
18: -0.9427476, 12.5802803, -0.8764961, 12.5771637, -10.7446365, 10.6645355
19: -5.3171000, 4.7437091, -5.2750196, 4.7425528, -7.6002617, 7.5510025
20: -3.4153533, 7.9842205, -3.3969731, 7.9762158, -10.2581139, 10.2818336
21: -1.9646441, 8.8846216, -1.9333327, 8.8805695, -8.9310303, 8.9209175
22: -9.2339993, 2.7863853, -9.2046986, 2.7840624, -8.6886120, 8.6767273
23: 1.3239465, 12.5102358, 1.3709530, 12.5102625, -7.7615414, 7.7057419
24: -2.7221792, 10.4991894, -2.6722882, 10.4969006, -8.2006874, 8.1422672
25: 0.3214939, 13.7655916, 0.3687716, 13.7633495, -9.4203739, 9.3741989
26: -17.4427547, 2.4890890, -17.3797112, 2.4824398, -14.6076660, 14.5597839
27: -10.3241587, 6.2966166, -10.2619753, 6.2912383, -9.1691151, 9.1086597
28: 1.0490863, 13.5818443, 1.1095617, 13.5810251, -9.6689148, 9.6028366
29: -5.1279945, 8.3608761, -5.0834131, 8.3570013, -8.6340084, 8.5961151
30: 5.9542885, 17.7160931, 5.9749947, 17.7135735, -7.6773243, 7.6684704
31: -3.4098532, 10.3971786, -3.3574493, 10.3937140, -9.1880341, 9.1382847
32: -19.6000862, -2.7043839, -19.5906410, -2.7343366, -10.7214165, 10.7442875
33: -47.0421677, -21.5037537, -47.0375023, -21.5627327, -14.5165024, 14.5893326
34: -29.7281456, -10.5639296, -29.7235088, -10.5836258, -10.6783600, 10.6688156
35: -29.2273254, -9.9331732, -29.2209625, -9.9472904, -10.7125168, 10.7044106
36: -31.8929367, -9.3874302, -31.8848839, -9.3847532, -12.7357788, 12.7052383
37: -46.1429062, -23.4788513, -46.1292267, -23.4941521, -16.0792313, 16.0587845
38: -34.3117218, -11.5099583, -34.3013000, -11.4927101, -15.1696091, 15.1039276
39: -56.3272324, -30.6280823, -56.3228073, -30.6717491, -13.1622086, 13.2195663
40: -40.2542114, -23.2798786, -40.2531738, -23.3000088, -8.1762619, 8.2157936
41: -26.7499237, -6.9564705, -26.7418900, -6.9760442, -11.2757034, 11.2729073
42: -14.5445757, -2.0928438, -14.5359840, -2.1202631, -8.5100689, 8.5322037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1681

## Relational analysis of IS_A2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0846268, upper bound: 5.0911179
time: 23.58 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0913255, upper bound: 5.0913254
time: 17.45 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 43.34 seconds
IS_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0910712, upper bound: 5.0695863
IS_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0912781, upper bound: 5.0762216
IS_A2_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0846268, upper bound: 5.0818151
IS_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0913255, upper bound: 5.0820123
IS_A2_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0846268, upper bound: 5.0859179
IS_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0913255, upper bound: 5.0861191
IS_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0806020, upper bound: 5.0911179
IS_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0872483, upper bound: 5.0913254
IS_A2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0846268, upper bound: 5.0836146
IS_A2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0913255, upper bound: 5.0838227
IS_A2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0757610, upper bound: 5.0911179
IS_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0824649, upper bound: 5.0913254
IS_A2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0846268, upper bound: 5.0911179
IS_A2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 43.34
Output dim: 11, lower bound: -5.0913255, upper bound: 5.0913254

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -20.2047672, 0.6664958, -20.2307854, 0.6920247, -13.3652725, 13.3921471
1: -6.4146752, 5.2579761, -6.4370174, 5.2598495, -6.3534298, 6.3812218
2: -11.0018368, 2.1890070, -11.0081263, 2.2063479, -8.5317535, 8.5067139
3: -12.3241425, 3.3637896, -12.3234987, 3.3963957, -11.1251373, 11.0958862
4: -22.0345688, -5.6836176, -22.0295887, -5.6868916, -9.1956062, 9.1854630
5: -10.7773438, 5.6006804, -10.7805471, 5.6333189, -12.1375885, 12.0983582
6: -22.4536133, -4.5431671, -22.4535370, -4.4954023, -11.1309624, 11.0770912
7: -9.5397949, 8.8794489, -9.5440607, 8.8865185, -12.4296036, 12.3991776
8: -26.3264656, -5.6302967, -26.3575058, -5.6314039, -9.7613373, 9.8046722
9: -14.5752649, 2.1140633, -14.5931988, 2.1638517, -12.9249954, 12.8882179
10: -5.8839931, 11.6872234, -5.9025497, 11.7163591, -13.1943665, 13.1903687
11: 9.6385660, 21.0913010, 9.5820599, 21.0946198, -7.4060383, 7.4684601
12: -15.1295795, 9.8405342, -15.1481495, 9.8657188, -18.4254684, 18.4673843
13: -28.0006046, -3.0970120, -27.9871483, -3.0330431, -12.8801041, 12.7940216
14: -31.3112030, 0.6331952, -31.3771305, 0.6313696, -21.3738251, 21.4666367
15: -24.8926964, -10.7274017, -24.9075603, -10.7071257, -8.8275146, 8.8409672
16: -6.9361792, 7.8636975, -6.9434443, 7.8789663, -10.0808792, 10.0813789
17: -14.6940317, 11.7384539, -14.7388306, 11.7284536, -21.6140137, 21.6690903
18: -0.8664925, 12.5659628, -0.9282351, 12.5722790, -10.6431732, 10.7077408
19: -5.2583113, 4.7349920, -5.2871933, 4.7363982, -7.5233440, 7.5567322
20: -3.3782561, 7.9553423, -3.3915675, 7.9637880, -10.2358513, 10.2065582
21: -1.9144418, 8.8610258, -1.9385130, 8.8523178, -8.8660736, 8.8845291
22: -9.1742954, 2.7593873, -9.1980934, 2.7731409, -8.6325455, 8.6403484
23: 1.4124026, 12.4925156, 1.3820753, 12.4919271, -7.6476460, 7.6874218
24: -2.6302910, 10.4768715, -2.6818345, 10.4805822, -8.0806198, 8.1397591
25: 0.4414601, 13.7276344, 0.4088883, 13.7260284, -9.2637615, 9.3050499
26: -17.3387203, 2.4713864, -17.3818188, 2.4662757, -14.4943771, 14.5423431
27: -10.2529058, 6.2789006, -10.3146038, 6.2753358, -9.0742035, 9.1344147
28: 1.1440294, 13.5711489, 1.1132882, 13.5683880, -9.5499458, 9.5906982
29: -5.0521078, 8.3429432, -5.0827408, 8.3430386, -8.5473785, 8.5814629
30: 6.0211887, 17.6940498, 5.9791846, 17.7026215, -7.6053810, 7.6429272
31: -3.3418379, 10.3797579, -3.3746450, 10.3812647, -9.1079330, 9.1335049
32: -19.5586147, -2.8006680, -19.5630035, -2.7396550, -10.6807442, 10.6095161
33: -47.0093880, -21.6183243, -47.0136986, -21.5338326, -14.5315971, 14.4418373
34: -29.7017517, -10.6158352, -29.7087975, -10.5807924, -10.6271591, 10.6294632
35: -29.1971588, -9.9883728, -29.2065315, -9.9304237, -10.6790733, 10.6548424
36: -31.8547020, -9.4339399, -31.8607922, -9.3763332, -12.6868591, 12.6440659
37: -46.0912933, -23.5121403, -46.0871849, -23.5009842, -15.9838257, 16.0066452
38: -34.2851486, -11.5428333, -34.2937317, -11.4966965, -15.1144104, 15.0873642
39: -56.3006516, -30.7147827, -56.3080978, -30.6358776, -13.1965790, 13.1044922
40: -40.2269783, -23.3241272, -40.2291183, -23.2967911, -8.1771660, 8.1381264
41: -26.7125950, -7.0253959, -26.7213078, -6.9906087, -11.2156239, 11.1912460
42: -14.5173883, -2.1314704, -14.5246592, -2.0965574, -8.5090046, 8.4742756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1378

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0834890, upper bound: 5.0690665
time: 14.64 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0909317, upper bound: 5.0690800
time: 10.87 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -20.2106228, 0.6721592, -20.2437172, 0.7032695, -13.4091492, 13.4090767
1: -6.4168472, 5.2713795, -6.4479079, 5.2829685, -6.3703880, 6.4005394
2: -11.0036907, 2.2036791, -11.0211086, 2.2317123, -8.5422668, 8.5282345
3: -12.3270597, 3.3790143, -12.3425636, 3.4233770, -11.1417847, 11.1298027
4: -22.0361443, -5.6570797, -22.0516701, -5.6411257, -9.2184944, 9.2340317
5: -10.7801342, 5.6172276, -10.7989893, 5.6618266, -12.1521606, 12.1311150
6: -22.4566879, -4.5191288, -22.4802017, -4.4557686, -11.1554756, 11.1289444
7: -9.5433865, 8.8994160, -9.5690994, 8.9208107, -12.4508667, 12.4438248
8: -26.3310604, -5.6175365, -26.3769302, -5.6099572, -9.7672615, 9.8347778
9: -14.5791531, 2.1251824, -14.6020193, 2.1863532, -12.9481888, 12.9075737
10: -5.8890686, 11.7044592, -5.9160433, 11.7488890, -13.2270279, 13.2227936
11: 9.6260080, 21.0918941, 9.5587921, 21.1038399, -7.4258060, 7.4814281
12: -15.1359301, 9.8459873, -15.1609592, 9.8789473, -18.4455490, 18.4836578
13: -28.0247631, -3.0918975, -28.0279350, -3.0015557, -12.9384117, 12.8181686
14: -31.3375587, 0.6359468, -31.4263840, 0.6546683, -21.4618988, 21.5115051
15: -24.9071426, -10.7226048, -24.9314976, -10.6888952, -8.8632889, 8.8687057
16: -6.9399729, 7.8809481, -6.9573278, 7.9081941, -10.1012268, 10.1082096
17: -14.7250214, 11.7394714, -14.7939339, 11.7553425, -21.6831436, 21.7255554
18: -0.8696527, 12.5688353, -0.9366813, 12.5783806, -10.6583405, 10.7194939
19: -5.2666521, 4.7365870, -5.3032169, 4.7396178, -7.5432644, 7.5757294
20: -3.3850672, 7.9637222, -3.4070656, 7.9784470, -10.2577209, 10.2328644
21: -1.9208694, 8.8766327, -1.9545457, 8.8786154, -8.8949852, 8.9161129
22: -9.1883230, 2.7610705, -9.2225075, 2.7832947, -8.6578865, 8.6601124
23: 1.3917148, 12.4931335, 1.3460472, 12.5023632, -7.6799183, 7.7197342
24: -2.6450973, 10.4781857, -2.7072973, 10.4913130, -8.1060257, 8.1549225
25: 0.4089439, 13.7286282, 0.3528061, 13.7506771, -9.3220882, 9.3294716
26: -17.3619728, 2.4739411, -17.4228134, 2.4829495, -14.5351257, 14.5821457
27: -10.2570076, 6.2854276, -10.3224754, 6.2865181, -9.0918999, 9.1533909
28: 1.1218278, 13.5713472, 1.0742457, 13.5765896, -9.5824089, 9.6252861
29: -5.0724640, 8.3442125, -5.1181731, 8.3566036, -8.5824699, 8.6112671
30: 6.0116005, 17.6949348, 5.9613523, 17.7087345, -7.6142769, 7.6608315
31: -3.3482647, 10.3819408, -3.3878853, 10.3856087, -9.1217842, 9.1521301
32: -19.5618896, -2.7792253, -19.5865211, -2.7018828, -10.7088699, 10.6565018
33: -47.0137215, -21.6004791, -47.0278168, -21.5014954, -14.5610390, 14.4759483
34: -29.7038498, -10.6042557, -29.7211819, -10.5604391, -10.6416893, 10.6549606
35: -29.2016487, -9.9823627, -29.2157822, -9.9204941, -10.6993027, 10.6715240
36: -31.8701973, -9.4275436, -31.8888130, -9.3616028, -12.7176704, 12.6718292
37: -46.1102982, -23.5085125, -46.1223755, -23.4779949, -16.0321579, 16.0407486
38: -34.2900314, -11.5356045, -34.3055649, -11.4840775, -15.1313858, 15.1047440
39: -56.3023834, -30.7044220, -56.3124809, -30.6171265, -13.2190781, 13.1224289
40: -40.2285080, -23.3159370, -40.2435989, -23.2830276, -8.1789932, 8.1571693
41: -26.7153740, -7.0084352, -26.7389679, -6.9613914, -11.2305717, 11.2268486
42: -14.5229502, -2.1296194, -14.5363350, -2.0925608, -8.5207367, 8.4917946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=87, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 663

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0837098, upper bound: 5.0759482
time: 24.92 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911381, upper bound: 5.0759482
time: 17.35 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2096977, 0.6615238, -20.2398262, 0.7015390, -13.4157028, 13.4236603
1: -6.4211721, 5.2698860, -6.4470921, 5.2821617, -6.3974743, 6.3966122
2: -11.0064163, 2.2121406, -11.0213337, 2.2320282, -8.5583000, 8.5254593
3: -12.3216190, 3.3756843, -12.3374081, 3.4234633, -11.1538696, 11.1113625
4: -22.0326271, -5.6534104, -22.0488663, -5.6403413, -9.2412758, 9.2110863
5: -10.7834578, 5.6288395, -10.7999878, 5.6621962, -12.1671371, 12.1283379
6: -22.4743690, -4.4767103, -22.4920998, -4.4538531, -11.1780205, 11.1608124
7: -9.5381031, 8.8945007, -9.5659399, 8.9222193, -12.4641953, 12.4184456
8: -26.3238525, -5.6257896, -26.3709431, -5.6085472, -9.7944756, 9.7985039
9: -14.5737801, 2.1161480, -14.5968294, 2.1866500, -12.9549942, 12.8920975
10: -5.8934250, 11.7012844, -5.9160581, 11.7491779, -13.2408104, 13.2102203
11: 9.6080894, 21.1027508, 9.5580511, 21.1088333, -7.4424391, 7.5087833
12: -15.1173019, 9.8308287, -15.1627769, 9.8669405, -18.4155121, 18.4792328
13: -28.0327606, -3.0851586, -28.0292015, -3.0001645, -12.9275589, 12.8436012
14: -31.3434830, 0.6381803, -31.4254494, 0.6543014, -21.4530334, 21.5513458
15: -24.9146671, -10.7166405, -24.9348183, -10.6887140, -8.8691559, 8.8610611
16: -6.9397078, 7.8764615, -6.9570417, 7.9079785, -10.1248817, 10.0960312
17: -14.7226810, 11.7229824, -14.7951813, 11.7456656, -21.6705780, 21.7257843
18: -0.8615177, 12.5660324, -0.9354451, 12.5746307, -10.6422157, 10.7251129
19: -5.2759519, 4.7345204, -5.3033972, 4.7407942, -7.5523453, 7.5782280
20: -3.3865161, 7.9690886, -3.4055111, 7.9786139, -10.2619705, 10.2382355
21: -1.9297664, 8.8774319, -1.9544140, 8.8800383, -8.9109726, 8.9070663
22: -9.1880112, 2.7647121, -9.2231817, 2.7838483, -8.6509647, 8.6683273
23: 1.3684818, 12.5026979, 1.3457074, 12.5085363, -7.7053185, 7.7320766
24: -2.6765802, 10.4921083, -2.7076337, 10.4986773, -8.1373711, 8.1764221
25: 0.3595848, 13.7524328, 0.3518565, 13.7642002, -9.3523140, 9.3808212
26: -17.3561993, 2.4713492, -17.4234467, 2.4786134, -14.5208588, 14.5882950
27: -10.2527113, 6.2906613, -10.3207245, 6.2886515, -9.0897865, 9.1593819
28: 1.1037385, 13.5772810, 1.0738742, 13.5811367, -9.5991096, 9.6361122
29: -5.0676041, 8.3437624, -5.1189728, 8.3556881, -8.5710220, 8.6240768
30: 5.9946146, 17.7093506, 5.9606867, 17.7158566, -7.6531487, 7.6669559
31: -3.3642883, 10.3831253, -3.3873885, 10.3899136, -9.1423111, 9.1481857
32: -19.5799656, -2.7487071, -19.5969238, -2.7016609, -10.7360115, 10.6899071
33: -47.0099030, -21.6144180, -47.0236511, -21.5015888, -14.5872383, 14.4513168
34: -29.6989021, -10.6091251, -29.7180481, -10.5600872, -10.6621284, 10.6436653
35: -29.2061024, -9.9859409, -29.2191544, -9.9202890, -10.6996384, 10.6820984
36: -31.8643265, -9.4174461, -31.8892021, -9.3612556, -12.7049026, 12.7017021
37: -46.1122513, -23.5134010, -46.1227798, -23.4815979, -16.0210953, 16.0504456
38: -34.2872391, -11.5312700, -34.3068047, -11.4837132, -15.1161652, 15.1155739
39: -56.2962875, -30.7170830, -56.3065643, -30.6168175, -13.2338600, 13.1058464
40: -40.2372360, -23.3086395, -40.2478027, -23.2829533, -8.1844749, 8.1641312
41: -26.7283382, -6.9826488, -26.7469368, -6.9608660, -11.2502308, 11.2471581
42: -14.5304337, -2.1297696, -14.5376453, -2.0926595, -8.5105782, 8.4970913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0818694
time: 22.88 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911879, upper bound: 5.0818770
time: 20.38 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2188454, 0.6766281, -20.2444592, 0.7020402, -13.4179077, 13.4428139
1: -6.4237490, 5.2758274, -6.4481993, 5.2823205, -6.3981171, 6.4035835
2: -11.0141487, 2.2247944, -11.0256681, 2.2325602, -8.5636406, 8.5417233
3: -12.3356524, 3.3932357, -12.3450317, 3.4241924, -11.1626434, 11.1362267
4: -22.0454140, -5.6390085, -22.0558128, -5.6397533, -9.2500725, 9.2327690
5: -10.7923841, 5.6433544, -10.8047495, 5.6628647, -12.1768723, 12.1482201
6: -22.4826546, -4.4755983, -22.4931526, -4.4548082, -11.1960030, 11.1662140
7: -9.5484791, 8.9068375, -9.5708065, 8.9226837, -12.4738464, 12.4350662
8: -26.3378849, -5.6129103, -26.3786602, -5.6083813, -9.7996788, 9.8186836
9: -14.5855103, 2.1320753, -14.6026154, 2.1880026, -12.9607391, 12.9140472
10: -5.8979521, 11.7147503, -5.9168868, 11.7508173, -13.2452927, 13.2276802
11: 9.5923805, 21.1091690, 9.5570650, 21.1122684, -7.4579077, 7.5071068
12: -15.1449308, 9.8540916, -15.1637726, 9.8798275, -18.4561081, 18.5006104
13: -28.0381889, -3.0799012, -28.0295334, -2.9981654, -12.9338913, 12.8542938
14: -31.3691502, 0.6458011, -31.4284172, 0.6581838, -21.4846878, 21.5520554
15: -24.9116707, -10.7148466, -24.9322815, -10.6878490, -8.8691864, 8.8810596
16: -6.9433460, 7.8866615, -6.9582100, 7.9082665, -10.1256142, 10.1077309
17: -14.7491570, 11.7470427, -14.7959251, 11.7586393, -21.7096176, 21.7432480
18: -0.8812239, 12.5752773, -0.9373128, 12.5793009, -10.6706657, 10.7299461
19: -5.2814369, 4.7392392, -5.3043904, 4.7409377, -7.5570831, 7.5855083
20: -3.3951244, 7.9767518, -3.4074969, 7.9790401, -10.2706566, 10.2508507
21: -1.9367669, 8.8825550, -1.9558653, 8.8802395, -8.9174500, 8.9178753
22: -9.1957283, 2.7647281, -9.2236462, 2.7845316, -8.6595535, 8.6749897
23: 1.3599095, 12.5051746, 1.3447721, 12.5085573, -7.7136726, 7.7349205
24: -2.6830399, 10.4942989, -2.7082887, 10.4990635, -8.1445084, 8.1796951
25: 0.3498101, 13.7556057, 0.3509500, 13.7645292, -9.3627281, 9.3846626
26: -17.3774357, 2.4802814, -17.4247055, 2.4834611, -14.5494843, 14.5944290
27: -10.2604027, 6.2923069, -10.3223610, 6.2887611, -9.1021080, 9.1625957
28: 1.0937936, 13.5805759, 1.0726836, 13.5812063, -9.6092758, 9.6407166
29: -5.0805273, 8.3491917, -5.1194687, 8.3579569, -8.5853596, 8.6250076
30: 5.9802122, 17.7100525, 5.9600334, 17.7157669, -7.6581154, 7.6694717
31: -3.3713977, 10.3902035, -3.3892965, 10.3900766, -9.1493645, 9.1605453
32: -19.5859222, -2.7413855, -19.5980072, -2.7007892, -10.7432976, 10.6979828
33: -47.0225296, -21.5905132, -47.0298805, -21.4999561, -14.5830040, 14.4783592
34: -29.7123280, -10.5968494, -29.7245522, -10.5595007, -10.6591568, 10.6610374
35: -29.2073669, -9.9761152, -29.2186165, -9.9196272, -10.7032661, 10.6873550
36: -31.8723927, -9.4170189, -31.8901634, -9.3606672, -12.7193832, 12.7014809
37: -46.1208725, -23.5065269, -46.1241226, -23.4790611, -16.0516815, 16.0557632
38: -34.2974052, -11.5296192, -34.3087196, -11.4833736, -15.1398087, 15.1170044
39: -56.3102798, -30.7010384, -56.3139801, -30.6159611, -13.2295189, 13.1265450
40: -40.2407341, -23.3031235, -40.2492218, -23.2826138, -8.2002792, 8.1637840
41: -26.7337189, -6.9768591, -26.7478638, -6.9603124, -11.2652969, 11.2490196
42: -14.5308037, -2.1249883, -14.5383987, -2.0919247, -8.5311356, 8.4972649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0859765
time: 23.79 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911879, upper bound: 5.0859836
time: 20.28 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.2152061, 0.6989119, -20.2260971, 0.6743736, -13.3985062, 13.4043083
1: -6.4230061, 5.2639098, -6.4377580, 5.2629867, -6.3837242, 6.3846149
2: -11.0095415, 2.2242966, -11.0116911, 2.1999612, -8.5184288, 8.5500526
3: -12.3195496, 3.4100904, -12.3241272, 3.3608775, -11.0899239, 11.1395569
4: -22.0229473, -5.6677194, -22.0379105, -5.6871800, -9.1629677, 9.2237053
5: -10.7816954, 5.6520762, -10.7903223, 5.6188593, -12.1159668, 12.1528587
6: -22.4669170, -4.4910564, -22.4762936, -4.5033965, -11.1322556, 11.1385345
7: -9.5378990, 8.8975925, -9.5540543, 8.8904152, -12.4057693, 12.4413147
8: -26.3351479, -5.6235218, -26.3463955, -5.6301212, -9.7892342, 9.7865677
9: -14.5804720, 2.1690347, -14.5855484, 2.1065631, -12.8576126, 12.9576187
10: -5.8920870, 11.7237396, -5.9053159, 11.6825695, -13.1612816, 13.2365265
11: 9.5659904, 21.1019287, 9.6112747, 21.1075611, -7.5034714, 7.4375420
12: -15.1545153, 9.8438663, -15.1263542, 9.8425922, -18.4758453, 18.4048767
13: -27.9977684, -3.0373271, -27.9987583, -3.0808561, -12.8130379, 12.8835945
14: -31.3786774, 0.6364522, -31.3245506, 0.6520100, -21.5002365, 21.3992996
15: -24.9088478, -10.7006073, -24.9242401, -10.7044439, -8.8180084, 8.8694763
16: -6.9356699, 7.8825464, -6.9489188, 7.8563385, -10.0761147, 10.1068172
17: -14.7558022, 11.7244911, -14.6834545, 11.7365618, -21.6980057, 21.5864868
18: -0.9319558, 12.5691223, -0.8533187, 12.5647812, -10.7248497, 10.6203232
19: -5.2986279, 4.7390366, -5.2603059, 4.7354722, -7.5713234, 7.5289268
20: -3.3968997, 7.9682288, -3.3809600, 7.9596090, -10.2177582, 10.2492561
21: -1.9465038, 8.8566780, -1.9195176, 8.8590317, -8.8869324, 8.8774071
22: -9.2077999, 2.7752059, -9.1822128, 2.7821589, -8.6528969, 8.6416225
23: 1.3627951, 12.4994106, 1.4013057, 12.5069962, -7.7189064, 7.6636467
24: -2.6946030, 10.4876719, -2.6502163, 10.4931450, -8.1683388, 8.1084690
25: 0.3813729, 13.7400570, 0.4126532, 13.7588968, -9.3551636, 9.3032417
26: -17.3983307, 2.4669054, -17.3339348, 2.4706321, -14.5549316, 14.4885025
27: -10.3137722, 6.2818065, -10.2496939, 6.2809825, -9.1451855, 9.0732155
28: 1.0913522, 13.5733490, 1.1428654, 13.5774040, -9.6205635, 9.5586166
29: -5.0901990, 8.3445721, -5.0490589, 8.3500147, -8.5935955, 8.5451393
30: 5.9738483, 17.7098846, 5.9996071, 17.7118206, -7.6538410, 7.6479759
31: -3.3934717, 10.3907871, -3.3432360, 10.3833714, -9.1545486, 9.1145687
32: -19.5744743, -2.7449775, -19.5808945, -2.7642210, -10.6643066, 10.6926918
33: -47.0213356, -21.5394955, -47.0202026, -21.6054859, -14.4536057, 14.5520363
34: -29.7087631, -10.5859241, -29.7076397, -10.6080189, -10.6342773, 10.6464119
35: -29.2161522, -9.9450483, -29.2137394, -9.9637623, -10.6856995, 10.6784706
36: -31.8604507, -9.4035320, -31.8591194, -9.3921528, -12.6982651, 12.6574249
37: -46.1051483, -23.5051594, -46.1009750, -23.5051403, -16.0361786, 15.9815140
38: -34.2950287, -11.5239487, -34.2845306, -11.5023317, -15.1457214, 15.0594826
39: -56.3150520, -30.6500473, -56.3069077, -30.6994991, -13.1212540, 13.1967239
40: -40.2376328, -23.2946510, -40.2477417, -23.3141251, -8.1560822, 8.1849518
41: -26.7307472, -6.9877634, -26.7334023, -6.9996910, -11.2366447, 11.2223053
42: -14.5303974, -2.0984416, -14.5291634, -2.1274176, -8.4916458, 8.4980183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0748996, upper bound: 5.0909665
time: 18.38 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0660132, upper bound: 5.0909792
time: 24.25 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2280712, 0.7100942, -20.2319298, 0.6800849, -13.4154472, 13.4481735
1: -6.4338770, 5.2870154, -6.4399490, 5.2763786, -6.4030037, 6.4015827
2: -11.0225544, 2.2496693, -11.0135155, 2.2146416, -8.5399437, 8.5604877
3: -12.3385944, 3.4370105, -12.3270741, 3.3761497, -11.1238556, 11.1561584
4: -22.0451069, -5.6219625, -22.0395164, -5.6606803, -9.2115173, 9.2466164
5: -10.8001575, 5.6805954, -10.7931423, 5.6353521, -12.1487503, 12.1673889
6: -22.4935627, -4.4513898, -22.4794426, -4.4793601, -11.1841202, 11.1630592
7: -9.5629492, 8.9318924, -9.5576210, 8.9103699, -12.4504013, 12.4625549
8: -26.3544044, -5.6021204, -26.3509350, -5.6173263, -9.8191452, 9.7924461
9: -14.5890656, 2.1915040, -14.5894756, 2.1176786, -12.8768387, 12.9807816
10: -5.9053268, 11.7563572, -5.9103665, 11.6997566, -13.1933250, 13.2692032
11: 9.5426893, 21.1111450, 9.5987530, 21.1081638, -7.5164490, 7.4573212
12: -15.1673880, 9.8570967, -15.1326723, 9.8480415, -18.4921417, 18.4250107
13: -28.0385227, -3.0058603, -28.0229282, -3.0758300, -12.8371658, 12.9419327
14: -31.4278812, 0.6597743, -31.3508186, 0.6547194, -21.5450974, 21.4874115
15: -24.9327106, -10.6824064, -24.9386501, -10.6996803, -8.8457489, 8.9052410
16: -6.9494820, 7.9117746, -6.9527111, 7.8735409, -10.1028404, 10.1271877
17: -14.8108749, 11.7514658, -14.7144251, 11.7375202, -21.7545395, 21.6555862
18: -0.9403801, 12.5751629, -0.8564796, 12.5676708, -10.7365417, 10.6354141
19: -5.3146877, 4.7422771, -5.2686677, 4.7370567, -7.5903397, 7.5488739
20: -3.4123812, 7.9828863, -3.3877690, 7.9679942, -10.2439766, 10.2711143
21: -1.9625249, 8.8829155, -1.9259657, 8.8745756, -8.9184151, 8.9063225
22: -9.2322083, 2.7853336, -9.1962357, 2.7838459, -8.6726341, 8.6669254
23: 1.3267672, 12.5098581, 1.3805976, 12.5076008, -7.7512169, 7.6959515
24: -2.7201090, 10.4983902, -2.6650276, 10.4944935, -8.1834717, 8.1339073
25: 0.3253167, 13.7647238, 0.3801498, 13.7598896, -9.3795624, 9.3616104
26: -17.4393177, 2.4836018, -17.3572235, 2.4731541, -14.5947113, 14.5292435
27: -10.3216696, 6.2929749, -10.2538252, 6.2875395, -9.1641159, 9.0910587
28: 1.0523102, 13.5815611, 1.1206996, 13.5776234, -9.6551933, 9.5911140
29: -5.1256576, 8.3581285, -5.0693932, 8.3513260, -8.6234341, 8.5802193
30: 5.9559669, 17.7159786, 5.9900131, 17.7127056, -7.6717110, 7.6567707
31: -3.4067292, 10.3951073, -3.3496780, 10.3855610, -9.1731529, 9.1284447
32: -19.5979691, -2.7071698, -19.5841484, -2.7428012, -10.7112694, 10.7208195
33: -47.0354385, -21.5071564, -47.0245972, -21.5876522, -14.4877129, 14.5814857
34: -29.7211723, -10.5655766, -29.7098198, -10.5964680, -10.6597900, 10.6609306
35: -29.2254410, -9.9350491, -29.2182674, -9.9577513, -10.7024155, 10.6987190
36: -31.8885574, -9.3888435, -31.8746471, -9.3857222, -12.7261543, 12.6881142
37: -46.1402817, -23.4821281, -46.1199341, -23.5015373, -16.0702286, 16.0298538
38: -34.3068466, -11.5114002, -34.2894402, -11.4950600, -15.1630402, 15.0764275
39: -56.3194199, -30.6313038, -56.3086433, -30.6891556, -13.1392097, 13.2191963
40: -40.2521591, -23.2808819, -40.2492828, -23.3059120, -8.1751366, 8.1867676
41: -26.7483997, -6.9585576, -26.7361813, -6.9827480, -11.2722454, 11.2372742
42: -14.5421515, -2.0944576, -14.5346909, -2.1255889, -8.5091820, 8.5097733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0797168, upper bound: 5.0911818
time: 22.65 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0871117, upper bound: 5.0911879
time: 21.04 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.2263794, 0.6931610, -20.2392464, 0.6942701, -13.4255676, 13.4307709
1: -6.4284744, 5.2827129, -6.4419732, 5.2822247, -6.4046497, 6.3919525
2: -11.0226059, 2.2360332, -11.0209026, 2.2263319, -8.5580025, 8.5388184
3: -12.3269644, 3.4007106, -12.3311386, 3.3926940, -11.1352768, 11.1186142
4: -22.0501328, -5.6330252, -22.0520821, -5.6472168, -9.2347374, 9.2313347
5: -10.7970190, 5.6580253, -10.7982159, 5.6490722, -12.1623230, 12.1419563
6: -22.4804840, -4.4729047, -22.4835663, -4.4784365, -11.1744537, 11.1572342
7: -9.5621090, 8.9219961, -9.5663109, 8.9219427, -12.4687424, 12.4339256
8: -26.3482990, -5.6057420, -26.3639984, -5.6052957, -9.8256950, 9.7855988
9: -14.5670156, 2.1346576, -14.5872097, 2.1313915, -12.8822403, 12.9108772
10: -5.8990712, 11.7160416, -5.9132485, 11.7108116, -13.2000122, 13.2237358
11: 9.5714684, 21.1022758, 9.5841093, 21.1081276, -7.4761105, 7.4651279
12: -15.1556826, 9.8631344, -15.1586819, 9.8695993, -18.4592361, 18.4726868
13: -28.0120296, -3.0813904, -28.0142517, -3.0730987, -12.8313446, 12.8481941
14: -31.3747139, 0.6401811, -31.3734665, 0.6513548, -21.4797363, 21.4971924
15: -24.9277191, -10.6947737, -24.9346619, -10.6990910, -8.8603096, 8.8898373
16: -6.9419785, 7.8869529, -6.9531598, 7.8830752, -10.1066589, 10.0994415
17: -14.7549763, 11.7346821, -14.7391300, 11.7455406, -21.6918106, 21.6655197
18: -0.8776600, 12.5508909, -0.8740535, 12.5613079, -10.6598625, 10.6494713
19: -5.2834244, 4.7326059, -5.2731562, 4.7367215, -7.5612812, 7.5472240
20: -3.4000084, 7.9778605, -3.3949230, 7.9749880, -10.2523460, 10.2510223
21: -1.9387970, 8.8791904, -1.9317074, 8.8779240, -8.9084015, 8.8939686
22: -9.2061138, 2.7811821, -9.2031975, 2.7820520, -8.6512337, 8.6698189
23: 1.3547453, 12.5012894, 1.3727530, 12.5062113, -7.7179070, 7.6986942
24: -2.6832376, 10.4803848, -2.6710432, 10.4881706, -8.1384315, 8.1294594
25: 0.3467700, 13.7583294, 0.3713164, 13.7606421, -9.3580360, 9.3655739
26: -17.3867378, 2.4719846, -17.3769989, 2.4731567, -14.5396729, 14.5480042
27: -10.2581024, 6.2760015, -10.2590990, 6.2801132, -9.0916710, 9.0845604
28: 1.0918553, 13.5747614, 1.1118672, 13.5774727, -9.6157341, 9.5955086
29: -5.0846734, 8.3498526, -5.0812273, 8.3517780, -8.5747299, 8.5856743
30: 5.9742861, 17.7083759, 5.9762673, 17.7099438, -7.6554871, 7.6509266
31: -3.3746786, 10.3859272, -3.3554585, 10.3875942, -9.1479530, 9.1219254
32: -19.5720444, -2.7423689, -19.5801048, -2.7364936, -10.6984940, 10.6828957
33: -47.0162430, -21.5731087, -47.0244102, -21.5656681, -14.4976730, 14.4909401
34: -29.7110996, -10.5904541, -29.7178974, -10.5846233, -10.6448669, 10.6318283
35: -29.2100277, -9.9639702, -29.2139130, -9.9486599, -10.6772423, 10.6813049
36: -31.8706703, -9.4145184, -31.8784466, -9.3862667, -12.6893730, 12.6792297
37: -46.1328354, -23.4974251, -46.1261292, -23.4963226, -16.0366669, 16.0580902
38: -34.2945175, -11.5251350, -34.2963409, -11.4941931, -15.1303101, 15.0912361
39: -56.3057861, -30.6898174, -56.3125076, -30.6741409, -13.1466408, 13.1334534
40: -40.2415581, -23.2983265, -40.2473259, -23.3006477, -8.1662369, 8.1780376
41: -26.7371483, -6.9749055, -26.7397957, -6.9773073, -11.2494373, 11.2351112
42: -14.5236816, -2.1229973, -14.5274458, -2.1211977, -8.4936924, 8.5003700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0836792
time: 19.29 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911880, upper bound: 5.0836844
time: 29.10 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -20.2193680, 0.6902566, -20.2152634, 0.6724803, -13.4064445, 13.3747101
1: -6.4239821, 5.2544785, -6.4202452, 5.2511520, -6.3782883, 6.3555145
2: -11.0088072, 2.2231712, -11.0098763, 2.2018900, -8.5204659, 8.5469265
3: -12.3260202, 3.4104500, -12.3330650, 3.3741345, -11.1108704, 11.1427803
4: -22.0290146, -5.6691279, -22.0441704, -5.6775255, -9.1806488, 9.2198830
5: -10.7817907, 5.6519518, -10.7897358, 5.6185598, -12.1170197, 12.1549797
6: -22.4455032, -4.4923830, -22.4452190, -4.5436163, -11.0742912, 11.1289597
7: -9.5428982, 8.8949909, -9.5585213, 8.8954859, -12.4162750, 12.4416351
8: -26.3424034, -5.6327090, -26.3388577, -5.6338639, -9.7976723, 9.7610703
9: -14.5850964, 2.1642566, -14.5812798, 2.1094162, -12.8676529, 12.9347153
10: -5.8920994, 11.7150898, -5.8890438, 11.6751165, -13.1616135, 13.2028236
11: 9.5655947, 21.0959244, 9.6143236, 21.0972919, -7.4905415, 7.4239941
12: -15.1489315, 9.8566999, -15.1393929, 9.8483295, -18.4728317, 18.4347534
13: -27.9939613, -3.0369742, -27.9958286, -3.0908134, -12.8102417, 12.8796425
14: -31.3807888, 0.6281133, -31.3213539, 0.6376743, -21.4836121, 21.3866043
15: -24.9056435, -10.7056084, -24.9065094, -10.7140045, -8.8289337, 8.8426800
16: -6.9363246, 7.8787737, -6.9414635, 7.8579903, -10.0821648, 10.0915260
17: -14.7554703, 11.7366095, -14.7013636, 11.7579002, -21.7092972, 21.6171494
18: -0.9329021, 12.5726128, -0.8653400, 12.5692520, -10.7192764, 10.6387367
19: -5.2985067, 4.7386336, -5.2611294, 4.7397623, -7.5765457, 7.5280361
20: -3.3960378, 7.9677820, -3.3820817, 7.9571409, -10.2167854, 10.2516327
21: -1.9469470, 8.8545189, -1.9182811, 8.8594189, -8.8934059, 8.8750038
22: -9.2057333, 2.7752137, -9.1828709, 2.7804060, -8.6546211, 8.6423244
23: 1.3625529, 12.4949217, 1.4062482, 12.5017281, -7.7177677, 7.6538124
24: -2.6946909, 10.4753628, -2.6307230, 10.4730587, -8.1543217, 8.0763149
25: 0.3809047, 13.7246342, 0.4356427, 13.7349911, -9.3397007, 9.2652817
26: -17.3977203, 2.4728384, -17.3506851, 2.4794443, -14.5586700, 14.5119095
27: -10.3133955, 6.2807665, -10.2497578, 6.2806311, -9.1342049, 9.0699577
28: 1.0921636, 13.5701542, 1.1421840, 13.5751009, -9.6200867, 9.5554161
29: -5.0879459, 8.3447323, -5.0546017, 8.3508186, -8.5859051, 8.5491486
30: 5.9734716, 17.6960812, 6.0127258, 17.6885948, -7.6405411, 7.6113338
31: -3.3953118, 10.3894701, -3.3475473, 10.3909235, -9.1649475, 9.1161613
32: -19.5513649, -2.7444468, -19.5434132, -2.8002486, -10.6044884, 10.6683865
33: -47.0207443, -21.5394440, -47.0202522, -21.5990086, -14.4579811, 14.5389519
34: -29.7051525, -10.5855026, -29.7013302, -10.6083107, -10.6287384, 10.6270981
35: -29.2040024, -9.9446678, -29.1938572, -9.9748077, -10.6570740, 10.6678848
36: -31.8378296, -9.4032421, -31.8252678, -9.4292459, -12.6362267, 12.6432114
37: -46.1052170, -23.5010204, -46.1103210, -23.4989433, -16.0365601, 16.0092239
38: -34.2780075, -11.5237427, -34.2592392, -11.5347691, -15.0931778, 15.0605011
39: -56.3136139, -30.6498356, -56.3044357, -30.7016983, -13.1142387, 13.1833191
40: -40.2336807, -23.2944202, -40.2410507, -23.3191185, -8.1401558, 8.1933403
41: -26.7127647, -6.9873877, -26.7046242, -7.0226727, -11.1902199, 11.2107620
42: -14.5283699, -2.0980563, -14.5231447, -2.1291175, -8.4814148, 8.5130634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0660132, upper bound: 5.0909665
time: 24.55 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0753858, upper bound: 5.0909792
time: 16.18 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.2322330, 0.7013979, -20.2211380, 0.6782153, -13.4234009, 13.4185677
1: -6.4348593, 5.2775617, -6.4224453, 5.2645731, -6.3975525, 6.3724842
2: -11.0217876, 2.2485466, -11.0117130, 2.2165637, -8.5419579, 8.5573750
3: -12.3450651, 3.4373827, -12.3360004, 3.3893776, -11.1447830, 11.1593895
4: -22.0511322, -5.6233397, -22.0457458, -5.6509838, -9.2291870, 9.2427750
5: -10.8002367, 5.6804376, -10.7925625, 5.6350756, -12.1497803, 12.1695251
6: -22.4722061, -4.4527411, -22.4483280, -4.5195646, -11.1261787, 11.1534882
7: -9.5679359, 8.9292564, -9.5620804, 8.9154453, -12.4609146, 12.4628716
8: -26.3616009, -5.6112461, -26.3433857, -5.6211085, -9.8276062, 9.7669449
9: -14.5936928, 2.1867657, -14.5852232, 2.1205118, -12.8869095, 12.9578857
10: -5.9053288, 11.7477217, -5.8940878, 11.6923513, -13.1936722, 13.2355194
11: 9.5422897, 21.1051292, 9.6017570, 21.0978889, -7.5035028, 7.4437828
12: -15.1617794, 9.8699493, -15.1457129, 9.8537188, -18.4891281, 18.4548645
13: -28.0347900, -3.0054984, -28.0199661, -3.0856841, -12.8343811, 12.9379807
14: -31.4299831, 0.6514516, -31.3476486, 0.6404891, -21.5284729, 21.4747391
15: -24.9295444, -10.6873970, -24.9209461, -10.7092190, -8.8566895, 8.8784466
16: -6.9501748, 7.9080162, -6.9452362, 7.8752027, -10.1089020, 10.1118584
17: -14.8105145, 11.7635412, -14.7323170, 11.7588577, -21.7657547, 21.6863022
18: -0.9413049, 12.5786180, -0.8685110, 12.5721540, -10.7309608, 10.6538277
19: -5.3145838, 4.7418613, -5.2694902, 4.7413664, -7.5955429, 7.5479794
20: -3.4115050, 7.9824300, -3.3889081, 7.9654975, -10.2430077, 10.2734566
21: -1.9629611, 8.8807611, -1.9247389, 8.8749619, -8.9249115, 8.9039459
22: -9.2301369, 2.7853460, -9.1968994, 2.7821054, -8.6743622, 8.6676559
23: 1.3265164, 12.5053692, 1.3855495, 12.5023508, -7.7500763, 7.6861401
24: -2.7201560, 10.4860954, -2.6455216, 10.4744167, -8.1694603, 8.1017456
25: 0.3248215, 13.7493143, 0.4031501, 13.7359915, -9.3640995, 9.3236618
26: -17.4387093, 2.4895113, -17.3739777, 2.4819961, -14.5984612, 14.5526810
27: -10.3213043, 6.2919645, -10.2538710, 6.2871995, -9.1531525, 9.0877953
28: 1.0531445, 13.5783682, 1.1199977, 13.5753222, -9.6546898, 9.5879173
29: -5.1233678, 8.3582687, -5.0749779, 8.3520966, -8.6157417, 8.5842209
30: 5.9556122, 17.7022057, 6.0031195, 17.6894493, -7.6584167, 7.6201496
31: -3.4085550, 10.3938131, -3.3539882, 10.3930779, -9.1835442, 9.1300182
32: -19.5748825, -2.7066514, -19.5466652, -2.7788081, -10.6514778, 10.6965313
33: -47.0348206, -21.5071259, -47.0245667, -21.5811729, -14.4920959, 14.5684319
34: -29.7176170, -10.5651636, -29.7035141, -10.5967522, -10.6542740, 10.6416321
35: -29.2132721, -9.9346504, -29.1983032, -9.9687710, -10.6737671, 10.6881332
36: -31.8658562, -9.3885689, -31.8407688, -9.4228115, -12.6640816, 12.6739159
37: -46.1402969, -23.4780273, -46.1292801, -23.4953804, -16.0706558, 16.0575790
38: -34.2897530, -11.5111904, -34.2640610, -11.5275078, -15.1104965, 15.0774651
39: -56.3179436, -30.6310024, -56.3061981, -30.6913834, -13.1321831, 13.2058067
40: -40.2481766, -23.2806282, -40.2426262, -23.3109436, -8.1592216, 8.1951485
41: -26.7303772, -6.9582067, -26.7074318, -7.0057025, -11.2258568, 11.2257233
42: -14.5401487, -2.0940678, -14.5287018, -2.1272953, -8.4989586, 8.5248241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1378

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0748800, upper bound: 5.0911818
time: 23.13 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0823265, upper bound: 5.0911879
time: 17.45 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.2198105, 0.6994283, -20.2352600, 0.6894639, -13.4176407, 13.4065208
1: -6.4241171, 5.2640901, -6.4403601, 5.2689152, -6.3906937, 6.3852673
2: -11.0138988, 2.2248144, -11.0194235, 2.2126660, -8.5346985, 8.5553894
3: -12.3271322, 3.4108481, -12.3381481, 3.3784444, -11.1147728, 11.1483078
4: -22.0299091, -5.6671591, -22.0507145, -5.6727886, -9.1846581, 9.2325020
5: -10.7864227, 5.6527381, -10.7992334, 5.6333108, -12.1358719, 12.1625824
6: -22.4678783, -4.4920197, -22.4845886, -4.5022860, -11.1376572, 11.1565170
7: -9.5427895, 8.8980885, -9.5644531, 8.9027548, -12.4223862, 12.4509811
8: -26.3428135, -5.6234179, -26.3604012, -5.6171732, -9.8094215, 9.7917709
9: -14.5862503, 2.1703200, -14.5972481, 2.1224625, -12.8795395, 12.9633942
10: -5.8929443, 11.7254057, -5.9098535, 11.6959944, -13.1787338, 13.2410011
11: 9.5650187, 21.1053696, 9.5955620, 21.1139908, -7.5017900, 7.4530125
12: -15.1555729, 9.8567543, -15.1539450, 9.8658791, -18.4972610, 18.4454651
13: -27.9981518, -3.0352602, -28.0041771, -3.0756626, -12.8237190, 12.8899155
14: -31.3815651, 0.6402829, -31.3502178, 0.6596746, -21.5009689, 21.4309540
15: -24.9063034, -10.6997547, -24.9212036, -10.7027321, -8.8380013, 8.8694992
16: -6.9368105, 7.8828621, -6.9525509, 7.8665133, -10.0878296, 10.1075630
17: -14.7565422, 11.7375364, -14.7099361, 11.7606401, -21.7155457, 21.6255569
18: -0.9338229, 12.5737944, -0.8730407, 12.5740347, -10.7297096, 10.6487885
19: -5.2996292, 4.7391706, -5.2657928, 4.7401910, -7.5785904, 7.5336952
20: -3.3988843, 7.9686403, -3.3895943, 7.9672966, -10.2303925, 10.2579727
21: -1.9479562, 8.8568697, -1.9265144, 8.8641415, -8.8977394, 8.8838844
22: -9.2082653, 2.7758839, -9.1899185, 2.7821572, -8.6595631, 8.6501808
23: 1.3618717, 12.4994564, 1.3927518, 12.5094557, -7.7217522, 7.6719780
24: -2.6952734, 10.4880447, -2.6566877, 10.4953232, -8.1716022, 8.1155968
25: 0.3804567, 13.7404070, 0.4029038, 13.7620687, -9.3590145, 9.3136482
26: -17.3995895, 2.4717698, -17.3551693, 2.4795291, -14.5610619, 14.5170898
27: -10.3154478, 6.2819033, -10.2574015, 6.2826557, -9.1483879, 9.0855370
28: 1.0901864, 13.5734091, 1.1328921, 13.5806999, -9.6251755, 9.5687943
29: -5.0906563, 8.3468504, -5.0619841, 8.3554668, -8.5945301, 8.5594749
30: 5.9731960, 17.7097855, 5.9851570, 17.7125702, -7.6563416, 7.6529140
31: -3.3953924, 10.3909092, -3.3503296, 10.3904209, -9.1668930, 9.1216469
32: -19.5756035, -2.7440856, -19.5868301, -2.7568769, -10.6723480, 10.6999283
33: -47.0276260, -21.5379333, -47.0328331, -21.5815849, -14.4806633, 14.5477982
34: -29.7152367, -10.5853300, -29.7210865, -10.5957880, -10.6516457, 10.6434250
35: -29.2156048, -9.9442692, -29.2150726, -9.9539738, -10.6909409, 10.6821060
36: -31.8613777, -9.4029551, -31.8671799, -9.3916836, -12.6980438, 12.6719170
37: -46.1065826, -23.5026302, -46.1095886, -23.4982071, -16.0414581, 16.0121155
38: -34.2969513, -11.5236044, -34.2946777, -11.5006180, -15.1471786, 15.0830994
39: -56.3225060, -30.6491585, -56.3208847, -30.6834106, -13.1419678, 13.1923637
40: -40.2390594, -23.2943573, -40.2511940, -23.3086090, -8.1557198, 8.2007675
41: -26.7316933, -6.9872012, -26.7387619, -6.9938970, -11.2384987, 11.2373734
42: -14.5311794, -2.0977297, -14.5295410, -2.1225996, -8.4918423, 8.5185356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0789254, upper bound: 5.0909665
time: 19.83 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0842516, upper bound: 5.0909792
time: 16.07 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2326584, 0.7106090, -20.2411137, 0.6951711, -13.4346085, 13.4503899
1: -6.4349933, 5.2871542, -6.4425216, 5.2823191, -6.4099617, 6.4022350
2: -11.0268707, 2.2501822, -11.0212688, 2.2273369, -8.5561981, 8.5658398
3: -12.3461971, 3.4377780, -12.3411064, 3.3936901, -11.1487198, 11.1649055
4: -22.0520592, -5.6213961, -22.0523033, -5.6462708, -9.2332039, 9.2553940
5: -10.8048754, 5.6812382, -10.8020411, 5.6498494, -12.1686401, 12.1771049
6: -22.4946136, -4.4523768, -22.4877281, -4.4782600, -11.1895447, 11.1810417
7: -9.5678101, 8.9323645, -9.5680161, 8.9226990, -12.4670334, 12.4722099
8: -26.3620872, -5.6019392, -26.3649120, -5.6044059, -9.8393211, 9.7976570
9: -14.5948429, 2.1928453, -14.6011906, 2.1335645, -12.8987656, 12.9865494
10: -5.9061818, 11.7579899, -5.9148836, 11.7132034, -13.2107697, 13.2737160
11: 9.5417223, 21.1145630, 9.5829916, 21.1145935, -7.5147629, 7.4727898
12: -15.1683693, 9.8700027, -15.1602859, 9.8713179, -18.5135498, 18.4655685
13: -28.0388813, -3.0037649, -28.0282974, -3.0705152, -12.8478546, 12.9482307
14: -31.4308281, 0.6635833, -31.3765430, 0.6624808, -21.5458221, 21.5190506
15: -24.9302025, -10.6815634, -24.9356651, -10.6979446, -8.8657532, 8.9052620
16: -6.9506464, 7.9121184, -6.9563622, 7.8837256, -10.1145554, 10.1279278
17: -14.8116636, 11.7643929, -14.7409372, 11.7616453, -21.7719879, 21.6946640
18: -0.9422324, 12.5797949, -0.8762102, 12.5769110, -10.7413864, 10.6638489
19: -5.3156943, 4.7423978, -5.2741394, 4.7417865, -7.5976048, 7.5536270
20: -3.4143324, 7.9833064, -3.3964114, 7.9756589, -10.2566261, 10.2797966
21: -1.9639926, 8.8831043, -1.9329736, 8.8796911, -8.9292297, 8.9127884
22: -9.2326689, 2.7860210, -9.2039528, 2.7838576, -8.6792946, 8.6754990
23: 1.3258433, 12.5099201, 1.3720381, 12.5100784, -7.7540684, 7.7042847
24: -2.7207758, 10.4987755, -2.6714799, 10.4966736, -8.1867466, 8.1410065
25: 0.3243747, 13.7650681, 0.3704104, 13.7630730, -9.3834114, 9.3720169
26: -17.4404869, 2.4884729, -17.3784180, 2.4821081, -14.6008606, 14.5578308
27: -10.3233538, 6.2930861, -10.2615156, 6.2892380, -9.1673393, 9.1033783
28: 1.0511391, 13.5816307, 1.1107252, 13.5809050, -9.6597786, 9.6012802
29: -5.1260986, 8.3604012, -5.0823288, 8.3567562, -8.6243629, 8.5945549
30: 5.9553690, 17.7158737, 5.9755974, 17.7134666, -7.6742249, 7.6617336
31: -3.4086356, 10.3952370, -3.3567731, 10.3926048, -9.1854935, 9.1355019
32: -19.5990982, -2.7062705, -19.5900726, -2.7354214, -10.7193336, 10.7280617
33: -47.0416603, -21.5055504, -47.0371971, -21.5637245, -14.5147552, 14.5772552
34: -29.7276306, -10.5649834, -29.7232132, -10.5842323, -10.6771851, 10.6579514
35: -29.2248802, -9.9343262, -29.2195854, -9.9479866, -10.7076645, 10.7023430
36: -31.8894787, -9.3882809, -31.8827057, -9.3852530, -12.7259102, 12.7025871
37: -46.1417007, -23.4796505, -46.1286087, -23.4946289, -16.0755463, 16.0604782
38: -34.3087006, -11.5110378, -34.2995605, -11.4933701, -15.1645355, 15.1000214
39: -56.3268318, -30.6303711, -56.3226013, -30.6731300, -13.1598969, 13.2148399
40: -40.2535706, -23.2805672, -40.2528076, -23.3004189, -8.1747856, 8.2025623
41: -26.7493553, -6.9579978, -26.7415924, -6.9769468, -11.2741203, 11.2523384
42: -14.5429478, -2.0937333, -14.5350637, -2.1207798, -8.5093555, 8.5303001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=86, inp2_unstable=88, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=7, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1478

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 641

## Relational analysis of IS_A2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0911818
time: 21.68 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0911880, upper bound: 5.0911879
time: 22.29 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 46.27 seconds
IS_A1_A1_A2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0834890, upper bound: 5.0690665
IS_A1_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0909317, upper bound: 5.0690800
IS_A1_A1_A2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0837098, upper bound: 5.0759482
IS_A1_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0911381, upper bound: 5.0759482
IS_A2_A1_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0818694
IS_A2_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0911879, upper bound: 5.0818770
IS_A2_A1_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0859765
IS_A2_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0911879, upper bound: 5.0859836
IS_A2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0748996, upper bound: 5.0909665
IS_A2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0660132, upper bound: 5.0909792
IS_A2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0797168, upper bound: 5.0911818
IS_A2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0871117, upper bound: 5.0911879
IS_A2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0836792
IS_A2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0911880, upper bound: 5.0836844
IS_A2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0660132, upper bound: 5.0909665
IS_A2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0753858, upper bound: 5.0909792
IS_A2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0748800, upper bound: 5.0911818
IS_A2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0823265, upper bound: 5.0911879
IS_A2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0789254, upper bound: 5.0909665
IS_A2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0842516, upper bound: 5.0909792
IS_A2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0911818
IS_A2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 46.27
Output dim: 11, lower bound: -5.0911880, upper bound: 5.0911879

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -20.2039700, 0.6662779, -20.2337208, 0.6998043, -13.3563805, 13.4039268
1: -6.4142642, 5.2578630, -6.4372139, 5.2704597, -6.3590431, 6.3836422
2: -11.0014620, 2.1889095, -11.0091887, 2.2211449, -8.5462341, 8.5051689
3: -12.3237000, 3.3636417, -12.3255577, 3.4119091, -11.1416931, 11.0954514
4: -22.0340290, -5.6837139, -22.0290794, -5.6637516, -9.2191582, 9.1795540
5: -10.7768974, 5.6006079, -10.7814264, 5.6500998, -12.1548080, 12.0955200
6: -22.4529209, -4.5432291, -22.4553795, -4.4810596, -11.1447601, 11.0770988
7: -9.5393944, 8.8793888, -9.5459185, 8.8984938, -12.4389954, 12.3971977
8: -26.3259869, -5.6303806, -26.3594856, -5.6147199, -9.7716408, 9.8026505
9: -14.5740490, 2.1138680, -14.5952950, 2.1750000, -12.9333267, 12.8902969
10: -5.8834839, 11.6869640, -5.9074001, 11.7360344, -13.2122269, 13.1939735
11: 9.6388111, 21.0906143, 9.5644035, 21.0944748, -7.4024229, 7.4861546
12: -15.1292858, 9.8400364, -15.1543064, 9.8679190, -18.4278412, 18.4688568
13: -28.0004654, -3.0973990, -27.9987659, -3.0295660, -12.8815460, 12.8053970
14: -31.3108158, 0.6325550, -31.3974037, 0.6320772, -21.3708649, 21.4919205
15: -24.8922577, -10.7276373, -24.9097214, -10.7029476, -8.8305740, 8.8438358
16: -6.9356203, 7.8635912, -6.9455662, 7.8907437, -10.0876236, 10.0855141
17: -14.6938801, 11.7380695, -14.7566185, 11.7289944, -21.6138306, 21.6877975
18: -0.8662174, 12.5649652, -0.9353564, 12.5745621, -10.6453667, 10.7179413
19: -5.2581887, 4.7344084, -5.2975354, 4.7365112, -7.5230808, 7.5688744
20: -3.3780105, 7.9552565, -3.3987367, 7.9678116, -10.2405090, 10.2133636
21: -1.9142652, 8.8609104, -1.9453707, 8.8582401, -8.8710518, 8.8935852
22: -9.1741743, 2.7588999, -9.2087517, 2.7740417, -8.6326180, 8.6505318
23: 1.4125429, 12.4920578, 1.3601875, 12.4915991, -7.6448765, 7.7097931
24: -2.6301434, 10.4759665, -2.6997526, 10.4810753, -8.0782661, 8.1592407
25: 0.4415631, 13.7268944, 0.3802922, 13.7261448, -9.2595692, 9.3339462
26: -17.3385201, 2.4708464, -17.3962650, 2.4677191, -14.4956512, 14.5580826
27: -10.2526855, 6.2775040, -10.3217487, 6.2763824, -9.0756683, 9.1434097
28: 1.1441903, 13.5705366, 1.0921147, 13.5680895, -9.5480042, 9.6125565
29: -5.0519772, 8.3425045, -5.0973048, 8.3438549, -8.5466919, 8.5962906
30: 6.0213661, 17.6933918, 5.9661460, 17.7030678, -7.6036491, 7.6550159
31: -3.3416770, 10.3789864, -3.3886218, 10.3815651, -9.1070290, 9.1478081
32: -19.5579205, -2.8007629, -19.5660248, -2.7290878, -10.6942101, 10.6110516
33: -47.0091782, -21.6185513, -47.0167580, -21.5219002, -14.5457611, 14.4416847
34: -29.7012444, -10.6159096, -29.7109432, -10.5714855, -10.6397629, 10.6280441
35: -29.1969910, -9.9885521, -29.2095337, -9.9251175, -10.6859398, 10.6558151
36: -31.8544445, -9.4346867, -31.8716774, -9.3746328, -12.6915588, 12.6523056
37: -46.0909958, -23.5136337, -46.0949783, -23.4995728, -15.9845734, 16.0155106
38: -34.2847672, -11.5436440, -34.2997589, -11.4926739, -15.1185226, 15.0920715
39: -56.3004646, -30.7152061, -56.3105316, -30.6320534, -13.2043762, 13.1010551
40: -40.2264557, -23.3241539, -40.2301750, -23.2926216, -8.1805534, 8.1379814
41: -26.7121010, -7.0254898, -26.7238827, -6.9785986, -11.2272987, 11.1934128
42: -14.5171871, -2.1315267, -14.5296221, -2.0945539, -8.5121822, 8.4789772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=86, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1378

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 923

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0900203, upper bound: 5.0642574
time: 24.00 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0902557, upper bound: 5.0684015
time: 8.68 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -20.2098522, 0.6719735, -20.2465019, 0.7110913, -13.3992233, 13.4206696
1: -6.4164405, 5.2712593, -6.4480624, 5.2936687, -6.3762817, 6.4021111
2: -11.0033016, 2.2035823, -11.0221748, 2.2464852, -8.5572548, 8.5266800
3: -12.3266315, 3.3788595, -12.3445930, 3.4387658, -11.1582184, 11.1293564
4: -22.0356140, -5.6571870, -22.0511837, -5.6180735, -9.2420464, 9.2281075
5: -10.7796879, 5.6171150, -10.7998343, 5.6785460, -12.1696167, 12.1282463
6: -22.4560680, -4.5191965, -22.4821091, -4.4414225, -11.1692772, 11.1290169
7: -9.5429821, 8.8993301, -9.5709248, 8.9326496, -12.4601746, 12.4418221
8: -26.3305721, -5.6176524, -26.3787861, -5.5932841, -9.7775497, 9.8324242
9: -14.5779810, 2.1249776, -14.6038952, 2.1975901, -12.9569473, 12.9092751
10: -5.8884935, 11.7042017, -5.9204464, 11.7685089, -13.2449226, 13.2259254
11: 9.6262531, 21.0911942, 9.5411425, 21.1035957, -7.4222412, 7.4994106
12: -15.1357460, 9.8454428, -15.1673326, 9.8812695, -18.4473572, 18.4853897
13: -28.0245476, -3.0923610, -28.0394878, -2.9982431, -12.9398270, 12.8294411
14: -31.3371201, 0.6353536, -31.4470730, 0.6557646, -21.4590530, 21.5365906
15: -24.9066525, -10.7228384, -24.9335289, -10.6849222, -8.8661728, 8.8715458
16: -6.9393578, 7.8808088, -6.9592776, 7.9199457, -10.1081848, 10.1114025
17: -14.7248669, 11.7390308, -14.8114662, 11.7558689, -21.6830139, 21.7440414
18: -0.8693788, 12.5678654, -0.9436660, 12.5805855, -10.6604576, 10.7294922
19: -5.2665267, 4.7360024, -5.3137178, 4.7397485, -7.5430698, 7.5878811
20: -3.3848176, 7.9636106, -3.4141958, 7.9824786, -10.2622795, 10.2387848
21: -1.9207315, 8.8765202, -1.9610302, 8.8844261, -8.8999329, 8.9246521
22: -9.1882191, 2.7605855, -9.2331476, 2.7841363, -8.6578274, 8.6702881
23: 1.3918363, 12.4926815, 1.3241713, 12.5020351, -7.6771507, 7.7420998
24: -2.6449440, 10.4772701, -2.7251923, 10.4917374, -8.1036530, 8.1743927
25: 0.4090593, 13.7278891, 0.3242424, 13.7507839, -9.3179188, 9.3583298
26: -17.3617592, 2.4733863, -17.4371490, 2.4843841, -14.5363960, 14.5977097
27: -10.2568064, 6.2840748, -10.3300419, 6.2875147, -9.0933609, 9.1620998
28: 1.1220195, 13.5707626, 1.0530808, 13.5762959, -9.5805016, 9.6471481
29: -5.0723562, 8.3437977, -5.1327124, 8.3573885, -8.5817451, 8.6261215
30: 6.0117583, 17.6942501, 5.9482555, 17.7091141, -7.6123962, 7.6723824
31: -3.3481104, 10.3810816, -3.4018900, 10.3857098, -9.1207619, 9.1662140
32: -19.5611916, -2.7793078, -19.5895615, -2.6913123, -10.7223206, 10.6580410
33: -47.0134964, -21.6007652, -47.0309105, -21.4895821, -14.5748520, 14.4757729
34: -29.7034111, -10.6043510, -29.7233429, -10.5511351, -10.6542549, 10.6535187
35: -29.2014523, -9.9825430, -29.2187557, -9.9152203, -10.7053185, 10.6726952
36: -31.8699608, -9.4282789, -31.9000435, -9.3598309, -12.7218742, 12.6795044
37: -46.1099358, -23.5099659, -46.1294708, -23.4769382, -16.0328369, 16.0493698
38: -34.2896690, -11.5363636, -34.3113823, -11.4802456, -15.1352997, 15.1097565
39: -56.3022156, -30.7048187, -56.3148842, -30.6131191, -13.2274094, 13.1188622
40: -40.2280350, -23.3159714, -40.2446823, -23.2788963, -8.1822968, 8.1570816
41: -26.7149048, -7.0085263, -26.7415485, -6.9494615, -11.2422333, 11.2290611
42: -14.5227499, -2.1297367, -14.5415010, -2.0905418, -8.5240097, 8.4965134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=87, inp2_unstable=86, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=7, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=22, inp2_unstable=22, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1478

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 923

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0902281, upper bound: 5.0710296
time: 7.50 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -5.0904630, upper bound: 5.0752872
time: 6.22 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 16.02 seconds
IS_A1_A1_A2_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 16.02
Output dim: 11, lower bound: -5.0900203, upper bound: 5.0642574
IS_A1_A1_A2_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 16.02
Output dim: 11, lower bound: -5.0902557, upper bound: 5.0684015
IS_A1_A1_A2_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 16.02
Output dim: 11, lower bound: -5.0902281, upper bound: 5.0710296
IS_A1_A1_A2_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 16.02
Output dim: 11, lower bound: -5.0904630, upper bound: 5.0752872
IS_A2_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0911879, upper bound: 5.0818770
IS_A2_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0911879, upper bound: 5.0859836
IS_A2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0748996, upper bound: 5.0909665
IS_A2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0660132, upper bound: 5.0909792
IS_A2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0797168, upper bound: 5.0911818
IS_A2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0871117, upper bound: 5.0911879
IS_A2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0911880, upper bound: 5.0836844
IS_A2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0660132, upper bound: 5.0909665
IS_A2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0753858, upper bound: 5.0909792
IS_A2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0748800, upper bound: 5.0911818
IS_A2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0823265, upper bound: 5.0911879
IS_A2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0789254, upper bound: 5.0909665
IS_A2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0842516, upper bound: 5.0909792
IS_A2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0837468, upper bound: 5.0911818
IS_A2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.02
Output dim: 11, lower bound: -5.0911880, upper bound: 5.0911879

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 29.51 + 1778.21 = 1807.72 seconds
