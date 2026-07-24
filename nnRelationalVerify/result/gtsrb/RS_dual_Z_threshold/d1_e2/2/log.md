## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 1800 seconds
Split limit: 100
Threshold: 6.3435205296


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9844437, 11.9844437)
1: (-9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8108845, 6.8108826)
2: (-4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8888779, 6.8888779)
3: (-13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6238213, 9.6238213)
4: (-5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9652634, 8.9652634)
5: (-8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5060806, 11.5060768)
6: (-24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8470535, 9.8470535)
7: (-9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9052200, 8.9052200)
8: (-12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4573116, 9.4573097)
9: (-7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8945656, 10.8945656)
10: (-7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5965004, 11.5965042)
11: (-4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2004929, 8.2004929)
12: (-16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4305344, 11.4305305)
13: (-21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4848022, 14.4848022)
14: (-22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8743286, 16.8743286)
15: (-9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8615570, 9.8615570)
16: (-9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1137238, 10.1137238)
17: (-20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7353363, 13.7353363)
18: (-3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1357079, 11.1357155)
19: (1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1972885, 9.1972885)
20: (-0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539)
21: (0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2512283, 12.2512283)
22: (1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3883705, 8.3883705)
23: (0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5089722, 9.5089722)
24: (-5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0550232, 12.0550270)
25: (-4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8815460, 11.8815460)
26: (2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147)
27: (0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4281807, 10.4281807)
28: (0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5216599, 11.5216637)
29: (-0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7304230, 6.7304211)
30: (-4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6928711, 12.6928635)
31: (-3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4107208, 11.4107246)
32: (-19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4112892, 9.4112892)
33: (-38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9177475, 15.9177475)
34: (-37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3373909, 10.3373909)
35: (-29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7973366, 11.7973328)
36: (-22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2297516, 9.2297516)
37: (-39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7198410, 15.7198410)
38: (-36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4880524, 14.4880486)
39: (-38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8117447, 14.8117447)
40: (-34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6939697, 8.6939678)
41: (-21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3942032, 12.3942032)
42: (-23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8710442, 9.8710442)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.66 + 20.37 = 23.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 26, lower bound: -6.3498704, upper bound: 6.3498704

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1686

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3461512, upper bound: 6.3490335
time: 21.04 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3490335, upper bound: 6.3461512
time: 24.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 45.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 45.43
Output dim: 26, lower bound: -6.3461512, upper bound: 6.3490335
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 45.43
Output dim: 26, lower bound: -6.3490335, upper bound: 6.3461512

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9850159, 11.9826508
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8112545, 6.8101578
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8894272, 6.8878136
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6239090, 9.6236191
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9655952, 8.9646568
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5060806, 11.5051994
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8468590, 9.8470154
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9055824, 8.9042702
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4587097, 9.4558544
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8944016, 10.8941727
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5963135, 11.5961609
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2002831, 8.2003822
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4285583, 11.4323044
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4840317, 14.4848137
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8740540, 16.8746185
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8619385, 9.8611145
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1137199, 10.1138420
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7348633, 13.7353439
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1354675, 11.1355057
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1972847, 9.1973267
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2509537, 12.2523384
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3880310, 8.3879929
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5095978, 9.5086517
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0550537, 12.0543900
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8815231, 11.8815346
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4277649, 10.4270477
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5216064, 11.5216255
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7304192, 6.7304192
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6927261, 12.6931229
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4107208, 11.4107323
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4109383, 9.4116364
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9171066, 15.9170265
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3371277, 10.3385086
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7971153, 11.7996254
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2297440, 9.2305737
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7196503, 15.7196426
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4876862, 14.4879036
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8116684, 14.8122253
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6939621, 8.6938686
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3940582, 12.3940125
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8712616, 9.8710327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1686

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 529

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3394993, upper bound: 6.3423202
time: 24.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3394379, upper bound: 6.3423816
time: 30.06 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9826508, 11.9844437
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8101559, 6.8108826
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8878136, 6.8888779
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6236191, 9.6238213
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9646568, 8.9652634
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5051956, 11.5060768
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8470535, 9.8468609
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9042702, 8.9052200
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4558525, 9.4573097
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8945656, 10.8944016
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5965004, 11.5963097
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2003822, 8.2004929
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4305344, 11.4285583
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4848022, 14.4840317
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8743286, 16.8740616
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8611145, 9.8615570
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1137238, 10.1137199
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7353363, 13.7348633
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1355057, 11.1357155
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1972885, 9.1972885
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2512283, 12.2509537
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3883705, 8.3880310
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5086517, 9.5089722
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0543823, 12.0550270
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8815384, 11.8815460
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4270477, 10.4281807
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5216599, 11.5216026
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7304192, 6.7304211
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6928711, 12.6927261
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4107208, 11.4107246
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4112892, 9.4109383
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9177475, 15.9171104
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3373909, 10.3371277
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7973366, 11.7971153
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2297516, 9.2297421
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7198410, 15.7196503
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4878998, 14.4880486
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8117447, 14.8116684
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6939697, 8.6939602
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3940125, 12.3942032
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8710327, 9.8710442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1686

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 529

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3423816, upper bound: 6.3394379
time: 19.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3423202, upper bound: 6.3394993
time: 27.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 48.99 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 48.99
Output dim: 26, lower bound: -6.3394993, upper bound: 6.3423202
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 48.99
Output dim: 26, lower bound: -6.3394379, upper bound: 6.3423816
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 48.99
Output dim: 26, lower bound: -6.3423816, upper bound: 6.3394379
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 48.99
Output dim: 26, lower bound: -6.3423202, upper bound: 6.3394993

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 23.03 + 151.03 = 174.06 seconds
