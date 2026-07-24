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
execution time: IAR + RelationalAnalysis = 2.49 + 20.33 = 22.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 26, lower bound: -6.3498704, upper bound: 6.3498704

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 529

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 658

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3491637, upper bound: 6.3464173
time: 12.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3464173, upper bound: 6.3491637
time: 10.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 22.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 22.41
Output dim: 26, lower bound: -6.3491637, upper bound: 6.3464173
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 22.41
Output dim: 26, lower bound: -6.3464173, upper bound: 6.3491637

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9781456, 11.9814873
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8107624, 6.8107605
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8888283, 6.8888321
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6156731, 9.6199188
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9590492, 8.9624023
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4987335, 11.5026398
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8440666, 9.8431339
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9015961, 8.9029427
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4550762, 9.4559517
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8843193, 10.8898201
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5820885, 11.5898590
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2004662, 8.2006035
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4285660, 11.4249992
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4856033, 14.4814148
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8727875, 16.8686676
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8549881, 9.8585281
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1072044, 10.1106682
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7297134, 13.7231369
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1336327, 11.1347389
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1968765, 9.1969604
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2511902, 12.2516785
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3870506, 8.3864517
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5051918, 9.5020256
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0526962, 12.0502205
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8785362, 11.8750114
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4262238, 10.4232292
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5198441, 11.5178070
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7293377, 6.7281094
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6917038, 12.6900177
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4106674, 11.4107590
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4094162, 9.4093704
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9172821, 15.9170532
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3341637, 10.3344002
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7957458, 11.7942047
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2278252, 9.2235985
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7167130, 15.7132950
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4871750, 14.4850502
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8096619, 14.8065033
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6848221, 8.6884899
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3941803, 12.3939247
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8708305, 9.8713531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 544

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3488878, upper bound: 6.3444187
time: 12.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3471641, upper bound: 6.3461411
time: 23.02 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9814873, 11.9781456
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8107586, 6.8107624
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8888321, 6.8888283
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6199188, 9.6156731
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9623985, 8.9590530
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5026398, 11.4987335
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8431320, 9.8440666
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9029427, 8.9015961
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4559498, 9.4550781
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8898201, 10.8843193
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5898552, 11.5820847
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2006035, 8.2004662
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4249954, 11.4285660
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4814148, 14.4855995
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8686676, 16.8727875
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8585281, 9.8549881
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1106682, 10.1072044
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7231369, 13.7297134
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1347389, 11.1336288
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1969604, 9.1968765
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2516785, 12.2511864
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3864517, 8.3870506
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5020256, 9.5051880
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0502243, 12.0526962
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8750114, 11.8785324
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4232330, 10.4262238
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5178070, 11.5198441
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7281094, 6.7293377
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6900253, 12.6917038
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4107590, 11.4106674
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4093704, 9.4094162
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9170532, 15.9172821
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3344002, 10.3341637
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7942047, 11.7957458
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2235985, 9.2278252
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7132950, 15.7167206
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4850540, 14.4871712
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8065033, 14.8096619
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6884918, 8.6848259
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3939285, 12.3941803
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8713531, 9.8708305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1415

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3461035, upper bound: 6.3475251
time: 11.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3447787, upper bound: 6.3488498
time: 9.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.69 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.69
Output dim: 26, lower bound: -6.3488878, upper bound: 6.3444187
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.69
Output dim: 26, lower bound: -6.3471641, upper bound: 6.3461411
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.69
Output dim: 26, lower bound: -6.3461035, upper bound: 6.3475251
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.69
Output dim: 26, lower bound: -6.3447787, upper bound: 6.3488498

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9736786, 11.9786377
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8102016, 6.8106556
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8879128, 6.8882294
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6152954, 9.6194725
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9591560, 8.9624786
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4990616, 11.5031738
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8431587, 9.8423595
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9016457, 8.9029999
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4553604, 9.4556828
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8846436, 10.8902168
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5811768, 11.5888557
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2004013, 8.2004318
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4286652, 11.4250793
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4876900, 14.4836273
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8735275, 16.8695831
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8551292, 9.8585968
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1066093, 10.1102066
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7289734, 13.7220154
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1338425, 11.1349373
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1969223, 9.1970100
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2512817, 12.2518387
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3860970, 8.3857765
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5044556, 9.5014648
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0489502, 12.0472679
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8770027, 11.8739433
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4249687, 10.4222450
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5192566, 11.5166435
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7289352, 6.7279911
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6912079, 12.6893234
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4081154, 11.4087524
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4080353, 9.4083138
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9141083, 15.9146271
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3332253, 10.3334999
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7937622, 11.7924538
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2248573, 9.2209702
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7149887, 15.7116776
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4830322, 14.4812050
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8039932, 14.8022385
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6869164, 8.6901016
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3930054, 12.3929939
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8706131, 9.8711662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 529

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 517

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3488120, upper bound: 6.3402051
time: 12.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3446729, upper bound: 6.3443429
time: 22.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9752998, 11.9770164
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8106556, 6.8102016
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8882217, 6.8879185
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6152267, 9.6195412
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9591293, 8.9625053
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4992676, 11.5029716
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8432922, 9.8422241
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9016571, 8.9029922
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4548073, 9.4562340
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8847160, 10.8901443
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5810852, 11.5889511
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2002907, 8.2005386
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4286423, 11.4251022
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4878120, 14.4835014
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8736954, 16.8694077
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8550568, 9.8586693
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1067429, 10.1100769
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7285843, 13.7223969
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1338272, 11.1349525
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1969261, 9.1970062
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2513428, 12.2517700
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3863754, 8.3854980
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5046234, 9.5012970
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0497437, 12.0464706
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8774681, 11.8734779
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4252357, 10.4219742
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5186768, 11.5172157
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7292175, 6.7277050
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6910095, 12.6895218
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4086647, 11.4082108
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4083595, 9.4079895
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9148483, 15.9138794
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3332634, 10.3334579
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7939911, 11.7922173
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2251968, 9.2206306
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7151031, 15.7115707
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4833221, 14.4809151
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8053970, 14.8008347
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6864357, 8.6905823
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3932495, 12.3927498
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8706398, 9.8711395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 947

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 722

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3342267, upper bound: 6.3459396
time: 35.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3469625, upper bound: 6.3332030
time: 40.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9739532, 11.9728355
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8094025, 6.8096752
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8900604, 6.8904953
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6192093, 9.6149559
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9630661, 8.9596176
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5030975, 11.4993591
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8406601, 9.8427067
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9021034, 8.9007187
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4541321, 9.4529076
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8891068, 10.8844376
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5865707, 11.5801468
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2002487, 8.2000809
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4256821, 11.4291000
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4771805, 14.4794235
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8662415, 16.8695679
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8577271, 9.8544083
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1149673, 10.1132240
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7176514, 13.7221680
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1347504, 11.1337051
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1966858, 9.1966019
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2516403, 12.2511139
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3869400, 8.3873940
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5011482, 9.5048218
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0463562, 12.0494232
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8746147, 11.8778496
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4234009, 10.4264450
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5160065, 11.5174484
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7267838, 6.7280579
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6901855, 12.6918030
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4091721, 11.4098434
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4056091, 9.4066353
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9117661, 15.9132462
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3292999, 10.3303185
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7924500, 11.7943077
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2262726, 9.2297745
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7083740, 15.7130051
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4909744, 14.4915085
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8048630, 14.8083115
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6832924, 8.6809559
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3877487, 12.3895836
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8673859, 9.8678780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 519

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3437762, upper bound: 6.3440826
time: 14.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3426624, upper bound: 6.3451975
time: 66.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9761772, 11.9706116
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8096733, 6.8094044
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8905029, 6.8900528
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6192017, 9.6149635
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9629669, 8.9597168
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5032654, 11.4991951
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8417740, 9.8415909
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9020653, 8.9007607
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4537811, 9.4532585
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8899384, 10.8836060
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5879211, 11.5788002
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2002182, 8.2001076
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4255371, 11.4292488
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4752350, 14.4813652
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8654556, 16.8703613
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8579521, 9.8541870
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1166878, 10.1114998
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7155838, 13.7242279
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1348190, 11.1336403
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1966820, 9.1966019
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2516098, 12.2511444
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3867950, 8.3875389
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5016670, 9.5043068
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0469437, 12.0488319
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8743248, 11.8781357
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4234543, 10.4263954
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5154114, 11.5180397
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7268295, 6.7280159
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6901169, 12.6918640
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4099350, 11.4090767
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4065895, 9.4056511
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9130249, 15.9119873
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3305511, 10.3290672
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7927628, 11.7939911
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2255516, 9.2304993
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7095871, 15.7117920
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4893951, 14.4930954
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8051529, 14.8080139
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6846199, 8.6796246
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3893280, 12.3880005
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8684006, 9.8668594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 529

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 519

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3424515, upper bound: 6.3454059
time: 12.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3413388, upper bound: 6.3465221
time: 14.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 29.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 26, lower bound: -6.3488120, upper bound: 6.3402051
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 26, lower bound: -6.3446729, upper bound: 6.3443429
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 26, lower bound: -6.3342267, upper bound: 6.3459396
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 26, lower bound: -6.3469625, upper bound: 6.3332030
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 26, lower bound: -6.3437762, upper bound: 6.3440826
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 26, lower bound: -6.3426624, upper bound: 6.3451975
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 26, lower bound: -6.3424515, upper bound: 6.3454059
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 26, lower bound: -6.3413388, upper bound: 6.3465221

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9677773, 11.9741974
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8094692, 6.8100739
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8880959, 6.8884525
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6141739, 9.6182442
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9568405, 8.9594040
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4998245, 11.5041733
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8370476, 9.8377647
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9008942, 8.9020042
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4528961, 9.4526634
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8826485, 10.8878021
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5811729, 11.5888481
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1993103, 8.1992416
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4263611, 11.4233398
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4846611, 14.4805679
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8743439, 16.8705444
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8545227, 9.8579903
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1054802, 10.1092644
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7282791, 13.7212677
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1317749, 11.1331825
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1966743, 9.1965714
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2507172, 12.2511482
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3863373, 8.3859367
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5029297, 9.5002632
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0460434, 12.0447388
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8765602, 11.8737030
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4229774, 10.4206085
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5173492, 11.5141144
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7272186, 6.7265587
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6911545, 12.6892548
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4048233, 11.4060364
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4048004, 9.4058838
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9117203, 15.9128265
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3298073, 10.3308792
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7923279, 11.7913742
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2230721, 9.2196312
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7108917, 15.7085876
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4804382, 14.4792557
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8023758, 14.8010254
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6839638, 8.6877060
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3886948, 12.3897476
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8668327, 9.8683243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 934

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 560

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3473098, upper bound: 6.3385860
time: 14.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3472368, upper bound: 6.3385860
time: 13.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9692383, 11.9727325
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8096218, 6.8099194
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8881378, 6.8884125
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6140671, 9.6183510
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9560776, 8.9601631
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5000610, 11.5039330
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8385620, 9.8362522
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9006500, 8.9022484
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4523392, 9.4532185
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8822289, 10.8882217
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5811729, 11.5888519
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1992111, 8.1993408
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4269333, 11.4227715
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4846306, 14.4805984
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8744812, 16.8704071
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8545189, 9.8579903
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1056671, 10.1090775
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7282257, 13.7213135
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1320877, 11.1328735
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1964836, 9.1967621
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2505875, 12.2512741
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3862572, 8.3860130
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5032578, 9.4999428
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0464172, 12.0443611
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8767586, 11.8735046
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4233360, 10.4202499
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5167236, 11.5147400
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7275009, 6.7262764
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6911392, 12.6892700
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4054031, 11.4054680
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4056015, 9.4050827
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9123154, 15.9122391
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3306046, 10.3300858
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7926788, 11.7910233
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2235146, 9.2191887
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7118988, 15.7075729
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4810791, 14.4786148
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8027725, 14.8006210
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6845207, 8.6871471
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3897629, 12.3886795
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8677711, 9.8673859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 563

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3152722, upper bound: 6.3148814
time: 11.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3152568, upper bound: 6.3148968
time: 32.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9617271, 11.9596825
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7993050, 6.7951622
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8750229, 6.8703537
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6089630, 9.6112175
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9472733, 8.9467621
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4962311, 11.4989090
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8266525, 9.8297081
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8877945, 8.8844872
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4217567, 9.4122906
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8708839, 10.8717880
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5738983, 11.5793610
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2005539, 8.2008629
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4420052, 11.4443054
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5008316, 14.4997177
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8712311, 16.8664551
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8501129, 9.8521271
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0917664, 10.0901756
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7312927, 13.7253189
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1261749, 11.1291885
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1956253, 9.1958504
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2440720, 12.2463226
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3650398, 8.3695202
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5090637, 9.5047264
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0419998, 12.0402298
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8604546, 11.8607330
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4258575, 10.4227562
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5247650, 11.5219574
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7156086, 6.7175083
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6830368, 12.6832809
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3861389, 11.3913269
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4089241, 9.4125671
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8935318, 15.8979263
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3150253, 10.3196793
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7707748, 11.7748184
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2019691, 9.2031593
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7198410, 15.7159195
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4596558, 14.4631844
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7845230, 14.7847900
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6989555, 8.7024212
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3959198, 12.3960724
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8691902, 9.8694649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 956

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3306660, upper bound: 6.3428197
time: 21.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3310969, upper bound: 6.3423886
time: 27.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9579620, 11.9634476
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7956161, 6.7988491
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8706589, 6.8747177
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6069031, 9.6132812
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9433899, 8.9506454
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4952087, 11.4999313
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8307762, 9.8255882
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8831520, 8.8891296
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4108696, 9.4231796
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8663597, 10.8763123
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5714951, 11.5817680
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2006187, 8.2008018
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4478493, 11.4384613
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5040283, 14.4965172
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8707504, 16.8669434
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8485146, 9.8537216
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0868416, 10.0951004
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7315063, 13.7250977
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1280670, 11.1272926
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1957741, 9.1957054
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2458878, 12.2444954
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3703957, 8.3641644
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5080566, 9.5057373
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0435028, 12.0387230
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8647194, 11.8564644
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4260254, 10.4225960
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5234222, 11.5233002
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7190266, 6.7140942
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6847687, 12.6815491
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3917770, 11.3856964
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4129372, 9.4085541
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8988876, 15.8925705
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3194847, 10.3152199
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7765961, 11.7689934
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2077255, 9.1974030
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7194519, 15.7163086
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4655914, 14.4572411
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7893524, 14.7799606
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6982727, 8.7031021
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3965759, 12.3954201
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8689651, 9.8696899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 641

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3464661, upper bound: 6.3309138
time: 22.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3442208, upper bound: 6.3326989
time: 19.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9730606, 11.9743958
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8093414, 6.8096027
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8900528, 6.8904781
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6193466, 9.6147881
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9640427, 8.9590721
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5030899, 11.4995193
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8394623, 9.8448143
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9024429, 8.9005241
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4544029, 9.4524612
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8892899, 10.8842621
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5862541, 11.5803299
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2000732, 8.1993523
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4253082, 11.4297600
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4786797, 14.4785309
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8663483, 16.8695221
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8576012, 9.8543091
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1144142, 10.1141663
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7177963, 13.7218094
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1344910, 11.1339684
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1969261, 9.1964607
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2519150, 12.2509499
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3873215, 8.3872414
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5009880, 9.5049019
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0460052, 12.0489388
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8747177, 11.8778076
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4232407, 10.4267654
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5166245, 11.5170822
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7265511, 6.7280579
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6901093, 12.6914673
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4091263, 11.4101181
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4049454, 9.4077911
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9112701, 15.9141388
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3285141, 10.3317299
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7922440, 11.7942619
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2262726, 9.2297745
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7076111, 15.7143707
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4911499, 14.4915085
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8048019, 14.8083267
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6823692, 8.6825809
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3868332, 12.3911972
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8665390, 9.8693733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 611

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 560

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3422493, upper bound: 6.3425021
time: 12.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3421900, upper bound: 6.3425622
time: 8.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9739532, 11.9719467
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8094025, 6.8096161
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8900604, 6.8904896
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6190414, 9.6149559
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9625244, 8.9596176
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5030975, 11.4993439
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8406601, 9.8415089
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9019089, 8.9007187
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4536858, 9.4529076
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8889313, 10.8844376
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5865707, 11.5798302
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2002487, 8.1999016
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4256821, 11.4287262
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4762840, 14.4794235
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8661957, 16.8695679
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8576279, 9.8544083
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1149673, 10.1126671
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7172928, 13.7221680
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1347504, 11.1334457
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1965446, 9.1966019
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2514725, 12.2511139
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3867874, 8.3873940
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5011482, 9.5046654
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0463562, 12.0490723
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8745728, 11.8778496
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4234009, 10.4262848
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5156326, 11.5174484
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7267838, 6.7278194
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6901855, 12.6917267
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4091721, 11.4097977
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4056091, 9.4059753
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9117661, 15.9127502
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3292999, 10.3295288
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7924500, 11.7941017
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2262726, 9.2297745
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7083740, 15.7122421
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4909744, 14.4915085
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8048630, 14.8082504
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6832924, 8.6800346
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3877487, 12.3886642
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8673859, 9.8670311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1783

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3403875, upper bound: 6.3442369
time: 23.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3417020, upper bound: 6.3442461
time: 20.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9752846, 11.9721718
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8096123, 6.8093319
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8904953, 6.8900375
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6193390, 9.6147957
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9639359, 8.9591751
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5032501, 11.4993553
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8405762, 9.8437004
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9024010, 8.9005623
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4540482, 9.4528122
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8901253, 10.8834305
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5876045, 11.5789833
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2000427, 8.1993828
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4251556, 11.4299088
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4767418, 14.4804764
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8655624, 16.8703156
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8578262, 9.8540840
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1161346, 10.1124420
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7157364, 13.7238770
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1345596, 11.1339035
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1969261, 9.1964607
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2518845, 12.2509842
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3871803, 8.3873863
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5014992, 9.5043869
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0466003, 12.0483513
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8744354, 11.8780937
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4232864, 10.4267120
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5160446, 11.5176735
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7265892, 6.7280159
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6900482, 12.6915359
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4098892, 11.4093513
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4059296, 9.4068108
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9125290, 15.9128799
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3297653, 10.3304787
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7925644, 11.7939453
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2255478, 9.2304955
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7088165, 15.7131577
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4895706, 14.4930878
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8050995, 14.8080368
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6836967, 8.6812477
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3884048, 12.3896141
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8675575, 9.8683548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 529

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 622

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3423186, upper bound: 6.3338557
time: 18.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3309009, upper bound: 6.3452736
time: 42.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9761772, 11.9697227
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8096733, 6.8093452
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8905029, 6.8900471
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6190338, 9.6149635
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9624214, 8.9597168
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5032654, 11.4991837
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8417740, 9.8403931
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9018707, 8.9007607
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4533348, 9.4532585
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8897629, 10.8836060
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5879211, 11.5784836
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2002182, 8.1999321
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4255371, 11.4288750
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4743462, 14.4813652
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8654099, 16.8703613
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8578491, 9.8541870
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1166878, 10.1109467
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7152328, 13.7242279
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1348190, 11.1333847
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1965408, 9.1966019
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2514420, 12.2511444
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3866425, 8.3875389
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5016670, 9.5041504
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0469437, 12.0484848
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8742828, 11.8781357
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4234543, 10.4262314
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5150528, 11.5180397
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7268295, 6.7277794
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6901169, 12.6917953
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4099350, 11.4090309
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4065895, 9.4049911
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9130249, 15.9114914
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3305511, 10.3282776
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7927628, 11.7937851
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2255478, 9.2304993
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7095871, 15.7110291
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4893875, 14.4930954
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8051529, 14.8079605
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6846199, 8.6787014
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3893280, 12.3870850
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8684006, 9.8660126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1528

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3393317, upper bound: 6.3461773
time: 13.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3409943, upper bound: 6.3445150
time: 18.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 33.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3473098, upper bound: 6.3385860
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3472368, upper bound: 6.3385860
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3152722, upper bound: 6.3148814
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3152568, upper bound: 6.3148968
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3306660, upper bound: 6.3428197
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3310969, upper bound: 6.3423886
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3464661, upper bound: 6.3309138
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3442208, upper bound: 6.3326989
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3422493, upper bound: 6.3425021
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3421900, upper bound: 6.3425622
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3403875, upper bound: 6.3442369
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3417020, upper bound: 6.3442461
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3423186, upper bound: 6.3338557
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3309009, upper bound: 6.3452736
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3393317, upper bound: 6.3461773
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.31
Output dim: 26, lower bound: -6.3409943, upper bound: 6.3445150

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9676895, 11.9755554
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8093052, 6.8099747
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8879700, 6.8883018
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6140175, 9.6173782
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9574585, 8.9589653
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4998093, 11.5041809
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8354568, 9.8376999
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9008064, 8.9012451
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4548817, 9.4521866
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8821373, 10.8867035
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5808449, 11.5877991
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1992416, 8.1992302
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4255676, 11.4232635
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4846458, 14.4804497
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8743362, 16.8705826
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8544922, 9.8576546
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1041451, 10.1085930
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7282181, 13.7209778
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1309967, 11.1332130
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1965981, 9.1964989
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2505417, 12.2511597
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3853340, 8.3856926
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5025024, 9.5001373
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0444183, 12.0446739
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8753319, 11.8737259
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4220390, 10.4206123
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5181503, 11.5140343
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7264175, 6.7267590
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6911392, 12.6892471
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4027634, 11.4059677
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4041977, 9.4064331
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9103851, 15.9124031
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3293190, 10.3307304
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7914810, 11.7910805
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2215080, 9.2194176
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7104034, 15.7085571
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4784698, 14.4785767
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8005142, 14.8004150
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6840248, 8.6876755
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3881073, 12.3897247
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8666992, 9.8683167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 923

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 524

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3471577, upper bound: 6.3377957
time: 13.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3465187, upper bound: 6.3384338
time: 23.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9677773, 11.9741135
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8093700, 6.8100739
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8879471, 6.8884525
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6133041, 9.6182442
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9564018, 8.9594040
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4998245, 11.5041580
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8370476, 9.8361721
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9001312, 8.9020042
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4524174, 9.4526634
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8815498, 10.8878021
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5801277, 11.5888481
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1992989, 8.1992416
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4263611, 11.4225426
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4846611, 14.4805527
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8743439, 16.8705368
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8541870, 9.8579903
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1048088, 10.1092644
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7279892, 13.7212677
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1317749, 11.1324043
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1966743, 9.1964951
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2507172, 12.2509804
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3863373, 8.3849335
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5029297, 9.4998322
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0460434, 12.0431137
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8765602, 11.8724670
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4229774, 10.4196739
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5172729, 11.5141144
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7272186, 6.7257538
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6911469, 12.6892548
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4048233, 11.4039764
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4048004, 9.4052773
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9117203, 15.9114952
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3298073, 10.3303871
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7923279, 11.7905273
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2230721, 9.2180672
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7108917, 15.7080994
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4804382, 14.4772949
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8023758, 14.7991638
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6839294, 8.6877060
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3886948, 12.3891602
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8668327, 9.8681908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 642

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3469423, upper bound: 6.3334574
time: 13.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3412131, upper bound: 6.3382070
time: 13.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9526901, 11.9592590
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7953758, 6.7986126
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8710861, 6.8751259
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6003609, 9.6084595
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9365540, 8.9451103
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4871140, 11.4936752
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8235207, 9.8180351
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8785591, 8.8855705
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4065914, 9.4198017
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8603706, 10.8717575
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5582123, 11.5713768
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2018700, 8.2024727
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4480515, 11.4379044
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5034180, 14.4938774
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8688583, 16.8645782
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8423271, 9.8488846
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0801964, 10.0902023
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7281723, 13.7209320
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1220894, 11.1225700
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1952744, 9.1950722
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2458496, 12.2444572
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3663635, 8.3591652
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4996567, 9.4946709
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0381012, 12.0314674
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8581696, 11.8479004
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4230118, 10.4180908
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5204468, 11.5193863
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7165527, 6.7108459
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6802139, 12.6760635
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3920860, 11.3859062
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4122276, 9.4078293
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8934021, 15.8862534
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3190842, 10.3149414
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7723999, 11.7636604
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2053223, 9.1934280
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7167053, 15.7123566
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4652557, 14.4567184
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7905502, 14.7785263
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6836662, 8.6915607
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3981476, 12.3966446
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8708344, 9.8723831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 611

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 619

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449173, upper bound: 6.3308507
time: 15.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3464030, upper bound: 6.3293639
time: 23.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9537773, 11.9581757
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7953796, 6.7986088
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8710670, 6.8751431
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6020813, 9.6067429
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9378510, 8.9438095
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4889603, 11.4918365
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8231926, 9.8183308
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8795929, 8.8845367
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4074879, 9.4189034
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8618050, 10.8703232
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5611038, 11.5684853
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2022858, 8.2020569
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4472885, 11.4386673
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5013885, 14.4959106
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8683777, 16.8650513
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8436737, 9.8475342
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0819435, 10.0884552
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7273331, 13.7217712
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1232491, 11.1213226
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1951408, 9.1952057
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2458572, 12.2444496
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3653984, 8.3601303
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4969864, 9.4973373
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0362396, 12.0333290
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8561554, 11.8499107
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4215088, 10.4195862
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5195084, 11.5203285
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7157745, 6.7116203
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6792755, 12.6769943
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3919945, 11.3859978
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4122124, 9.4078445
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8925705, 15.8870850
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3192024, 10.3148232
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7712631, 11.7647934
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2037506, 9.1949997
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7154922, 15.7135086
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4650650, 14.4569016
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7879181, 14.7811584
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6867332, 8.6884956
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3977966, 12.3969994
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8716545, 9.8715591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1418

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3437808, upper bound: 6.3293271
time: 33.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3408489, upper bound: 6.3322589
time: 20.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9744110, 11.9695244
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8087826, 6.8093433
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8897285, 6.8903618
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6116333, 9.6116638
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9548302, 8.9563026
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4983749, 11.4973145
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8404999, 9.8411846
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8948250, 8.8977089
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4485435, 9.4538479
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8770905, 10.8792496
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5842171, 11.5779381
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1964912, 8.1961708
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4232941, 11.4218559
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4749718, 14.4780273
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8677063, 16.8688126
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8563004, 9.8533020
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1117592, 10.1112328
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7177505, 13.7194214
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1294060, 11.1283150
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1951180, 9.1956711
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2513657, 12.2524796
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3856201, 8.3846817
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4974060, 9.4967346
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0414734, 12.0378952
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8727798, 11.8736725
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4212227, 10.4238548
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5155029, 11.5187950
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7240944, 6.7215328
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6894836, 12.6902466
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4056854, 11.4026527
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4054642, 9.4043922
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9082642, 15.9045715
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3289871, 10.3219147
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7877274, 11.7826576
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2232246, 9.2227936
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7013321, 15.6960678
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4884949, 14.4861679
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8009186, 14.7991791
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6832581, 8.6799736
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3843231, 12.3808365
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8672295, 9.8663216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3281035, upper bound: 6.3297572
time: 30.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3272316, upper bound: 6.3306292
time: 24.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9715309, 11.9724045
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8091259, 6.8089962
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8899269, 6.8901615
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6157455, 9.6075478
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9592094, 8.9519196
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5010681, 11.4946251
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8403358, 9.8413486
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8988991, 8.8936386
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4546242, 9.4477615
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8837433, 10.8725967
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5846672, 11.5774879
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1965141, 8.1961479
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4188080, 11.4263344
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4748955, 14.4781113
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8654175, 16.8711014
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8565178, 9.8530846
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1135330, 10.1094589
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7145538, 13.7226257
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1296196, 11.1281013
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1956177, 9.1951714
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2528458, 12.2509995
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3840714, 8.3862305
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4932175, 9.5009155
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0351791, 12.0441933
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8703918, 11.8760605
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4209785, 10.4241066
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5169830, 11.5173187
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7205009, 6.7251263
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6886978, 12.6910248
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4020233, 11.4063110
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4040222, 9.4058342
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9035873, 15.9092407
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3216858, 10.3292160
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7810059, 11.7893791
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2192879, 9.2267342
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6921997, 15.7052002
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4856262, 14.4890366
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7957916, 14.8043137
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6832275, 8.6800041
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3799133, 12.3852501
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8666840, 9.8668671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1418

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3412620, upper bound: 6.3408741
time: 21.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3383301, upper bound: 6.3438061
time: 9.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9714279, 11.9678879
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8059845, 6.8046513
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8896217, 6.8890438
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6129456, 9.6062775
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9575081, 8.9506149
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5001526, 11.4956093
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8505478, 9.8563957
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8985710, 8.8954582
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4505711, 9.4480419
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8791504, 10.8687668
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5862885, 11.5774918
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1947403, 8.1954002
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4271889, 11.4322777
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4602890, 14.4585724
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8646164, 16.8694077
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8567543, 9.8524094
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1157303, 10.1118927
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7159576, 13.7246704
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1276970, 11.1287537
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1963348, 9.1960144
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2580414, 12.2559967
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3863525, 8.3867626
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4952812, 9.4996262
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0367050, 12.0409241
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8729248, 11.8769608
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4130936, 10.4190521
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5129318, 11.5151558
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7193947, 6.7226124
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6805267, 12.6843872
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4040222, 11.4049416
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4082603, 9.4093704
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9144363, 15.9146423
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3220520, 10.3245239
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7962494, 11.7976265
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2247429, 9.2293758
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7035675, 15.7086487
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4950333, 14.4961472
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8037491, 14.8037567
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6853981, 8.6839275
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3961029, 12.3993492
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8648148, 9.8661423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 706

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1446

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3281940, upper bound: 6.3444590
time: 32.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3300863, upper bound: 6.3425670
time: 11.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9755783, 11.9693298
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8092995, 6.8076878
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8902702, 6.8889885
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6172752, 9.6121674
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9609985, 8.9572105
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5027466, 11.4982758
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8405571, 9.8401623
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8998909, 8.8976135
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4496803, 9.4472580
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8876076, 10.8802109
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5876465, 11.5785751
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1981316, 8.1987495
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4238052, 11.4281654
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4720459, 14.4776802
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8653641, 16.8703308
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8576050, 9.8537483
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1162949, 10.1106644
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7150192, 13.7240753
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1299210, 11.1303825
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1950035, 9.1956177
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2504120, 12.2505608
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3855972, 8.3869495
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5002174, 9.5033302
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0441971, 12.0469284
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8736649, 11.8778152
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4226532, 10.4257622
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5150986, 11.5176430
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7246933, 6.7264977
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6900635, 12.6917419
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4045029, 11.4055939
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4060898, 9.4051704
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9119339, 15.9109116
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3285484, 10.3271217
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7903137, 11.7923088
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2253113, 9.2305222
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7064514, 15.7090607
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4893417, 14.4930573
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8050461, 14.8079147
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6841507, 8.6791477
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3886490, 12.3871689
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8680954, 9.8653183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 639

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3340644, upper bound: 6.3404643
time: 12.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3336182, upper bound: 6.3409103
time: 9.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9757843, 11.9691238
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8080139, 6.8089733
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8894424, 6.8898163
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6162338, 9.6132088
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9599037, 8.9583092
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5023575, 11.4986610
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8415413, 9.8391800
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8987274, 8.8987770
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4473381, 9.4496002
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8863678, 10.8814507
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5879974, 11.5782280
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1990356, 8.1978493
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4248276, 11.4271469
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4706802, 14.4790459
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8653641, 16.8703308
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8574066, 9.8539429
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1164055, 10.1105537
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7150803, 13.7240143
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1318207, 11.1284828
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1955605, 9.1950645
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2508621, 12.2501144
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3860474, 8.3864994
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5008430, 9.5027084
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0453873, 12.0457420
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8739624, 11.8775177
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4229889, 10.4254265
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5146561, 11.5180779
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7255554, 6.7256413
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6900482, 12.6917496
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4065018, 11.4035988
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4067650, 9.4044991
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9124527, 15.9103851
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3294067, 10.3262634
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7912827, 11.7913361
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2255707, 9.2302628
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7076111, 15.7078934
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4893570, 14.4930420
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8051147, 14.8078461
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6850739, 8.6782284
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3894119, 12.3864136
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8677063, 9.8657074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 659

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1418

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3405543, upper bound: 6.3411431
time: 21.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3376224, upper bound: 6.3440750
time: 20.81 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 44.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3471577, upper bound: 6.3377957
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3465187, upper bound: 6.3384338
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3469423, upper bound: 6.3334574
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3412131, upper bound: 6.3382070
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3449173, upper bound: 6.3308507
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3464030, upper bound: 6.3293639
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3437808, upper bound: 6.3293271
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3408489, upper bound: 6.3322589
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3281035, upper bound: 6.3297572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3272316, upper bound: 6.3306292
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3412620, upper bound: 6.3408741
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3383301, upper bound: 6.3438061
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3281940, upper bound: 6.3444590
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3300863, upper bound: 6.3425670
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3340644, upper bound: 6.3404643
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3336182, upper bound: 6.3409103
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3405543, upper bound: 6.3411431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.51
Output dim: 26, lower bound: -6.3376224, upper bound: 6.3440750

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9623184, 11.9721222
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8074265, 6.8087063
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8896637, 6.8906326
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6153259, 9.6187668
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9574928, 8.9589996
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5018921, 11.5068207
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8356094, 9.8381004
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9019051, 8.9026184
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4549484, 9.4522438
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8825760, 10.8877106
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5802345, 11.5885506
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2000122, 8.2003021
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4286728, 11.4257050
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4819145, 14.4763718
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8730087, 16.8691254
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8540382, 9.8573380
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1077995, 10.1133003
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7229462, 13.7139587
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1298332, 11.1322060
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1964874, 9.1964111
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2502441, 12.2509613
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3864899, 8.3866348
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5003242, 9.4985123
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0404968, 12.0417328
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8756790, 11.8739281
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4223747, 10.4210320
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5176849, 11.5130196
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7252922, 6.7257786
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6921692, 12.6899948
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4008484, 11.4045258
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4025230, 9.4051132
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9097137, 15.9118118
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3261528, 10.3281631
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7909431, 11.7906342
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2247467, 9.2221127
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7100449, 15.7082901
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4833603, 14.4824219
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8007812, 14.8007431
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6843147, 8.6879959
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3846741, 12.3871460
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8644524, 9.8665466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3456732, upper bound: 6.3263849
time: 11.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3357463, upper bound: 6.3363112
time: 11.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9642563, 11.9701843
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8080368, 6.8080978
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8903008, 6.8899975
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6154060, 9.6186829
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9574928, 8.9589996
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5024490, 11.5062599
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8358574, 9.8378525
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9021835, 8.9023399
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4549408, 9.4522533
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8831444, 10.8871422
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5815926, 11.5871887
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2003136, 8.1999969
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4280090, 11.4263687
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4805641, 14.4777184
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8728790, 16.8692551
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8541794, 9.8572006
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1088524, 10.1122437
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7211914, 13.7157059
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1299858, 11.1320457
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1965103, 9.1963882
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2503510, 12.2508545
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3862762, 8.3868523
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5008736, 9.4979591
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0414734, 12.0407562
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8755264, 11.8740845
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4224663, 10.4209442
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5171356, 11.5135765
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7254372, 6.7256336
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6918869, 12.6902771
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4013290, 11.4040489
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4028778, 9.4047623
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9097977, 15.9117355
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3267479, 10.3275681
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7910347, 11.7905426
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2242050, 9.2226505
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7101288, 15.7081985
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4823227, 14.4834671
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8008423, 14.8006821
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6843452, 8.6879654
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3855286, 12.3862915
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8649254, 9.8660736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 529

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 518

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3464628, upper bound: 6.3362770
time: 21.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3429104, upper bound: 6.3382432
time: 18.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9525070, 11.9623528
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8047314, 6.8059750
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8840370, 6.8853722
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6042213, 9.6112709
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9437447, 8.9498482
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4903717, 11.4969673
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8359184, 9.8345585
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8920860, 8.8957443
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4432316, 9.4456654
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8653259, 10.8755493
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5601730, 11.5737877
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2001572, 8.2002563
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4212570, 11.4147072
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4914322, 14.4839401
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8730240, 16.8679733
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8443184, 9.8505402
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0901108, 10.0980721
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7181473, 13.7082367
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1311455, 11.1327400
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1971054, 9.1969414
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2509079, 12.2512589
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3817635, 8.3791103
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4950180, 9.4897461
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0417862, 12.0374794
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8677368, 11.8607788
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4174576, 10.4123383
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5138321, 11.5096359
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7241020, 6.7216263
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6863861, 12.6829529
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4051208, 11.4041710
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4054184, 9.4056778
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9067459, 15.9051476
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3299904, 10.3304405
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7869263, 11.7833710
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2130890, 9.2045975
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7062531, 15.7021484
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4740601, 14.4688568
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7950897, 14.7891006
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6694069, 8.6763821
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3893738, 12.3896255
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8676147, 9.8690720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 756

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3350063, upper bound: 6.3323772
time: 15.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3458655, upper bound: 6.3215194
time: 17.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9605713, 11.9647064
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7947121, 6.7986679
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8689232, 6.8735752
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.5951805, 9.6032944
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9346008, 8.9467392
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4835281, 11.4904404
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8232918, 9.8169861
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8673058, 8.8770180
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4127769, 9.4309559
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8462677, 10.8599358
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5600815, 11.5728378
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1952820, 8.1959763
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4495735, 11.4368706
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5036659, 14.4940872
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8699341, 16.8658295
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8425598, 9.8492050
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0749168, 10.0862045
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7290115, 13.7218704
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1140900, 11.1163673
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1953583, 9.1951714
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2474594, 12.2464371
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3650665, 8.3574944
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4902878, 9.4823418
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0285950, 12.0186844
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8550720, 11.8437195
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4216614, 10.4169312
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5276337, 11.5284348
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7116241, 6.7045021
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6791763, 12.6746674
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3822479, 11.3743172
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4095268, 9.4027748
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8804550, 15.8688049
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3087006, 10.2985916
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7587547, 11.7451973
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.1958466, 9.1807938
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6975174, 15.6871414
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4541931, 14.4423485
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7763977, 14.7596817
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6804848, 8.6875038
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3862305, 12.3812370
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8678703, 9.8683434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1418

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3444773, upper bound: 6.3274788
time: 13.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415469, upper bound: 6.3304107
time: 14.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9581375, 11.9671402
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7954292, 6.7979488
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8695335, 6.8729649
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.5951958, 9.6032791
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9381790, 8.9431572
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4838791, 11.4900894
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8224716, 9.8178082
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8700066, 8.8743172
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4177437, 9.4259892
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8485489, 10.8576546
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5596695, 11.5732460
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1953773, 8.1958847
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4470177, 11.4394264
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5036278, 14.4941292
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8701172, 16.8656464
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8426476, 9.8491173
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0761986, 10.0849228
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7291183, 13.7217712
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1158905, 11.1145630
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1953773, 9.1951523
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2478333, 12.2460670
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3646927, 8.3578682
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4873199, 9.4853058
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0253296, 12.0219536
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8539886, 11.8448067
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4218597, 10.4167366
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5294952, 11.5265732
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7102127, 6.7059135
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6788177, 12.6750259
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3804932, 11.3760757
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4071732, 9.4051285
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8759537, 15.8733063
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3027344, 10.3045578
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7539330, 11.7500229
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.1926880, 9.1839523
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6914978, 15.6931610
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4508972, 14.4456520
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7717056, 14.7643814
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6796074, 8.6883812
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3827362, 12.3847351
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8667946, 9.8694191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 524

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3462507, upper bound: 6.3285735
time: 30.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3456147, upper bound: 6.3292118
time: 20.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9483109, 11.9537735
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7946796, 6.7980118
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8700714, 6.8743477
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6002388, 9.6054382
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9357796, 8.9421883
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4890976, 11.4925308
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8210068, 9.8158798
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8794441, 8.8843765
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4068661, 9.4184151
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8585434, 10.8680000
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5571709, 11.5653687
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.2009010, 8.1999626
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4432526, 11.4338837
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5013962, 14.4959183
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8658829, 16.8620377
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8395996, 9.8443146
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0819397, 10.0884628
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7246094, 13.7183228
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1237411, 11.1214066
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1950226, 9.1950150
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2451477, 12.2435188
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3653412, 8.3602562
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4969940, 9.4973831
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0361176, 12.0331841
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8562088, 11.8499298
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4227867, 10.4204369
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5183945, 11.5188828
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7158356, 6.7116661
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6781082, 12.6753998
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3919258, 11.3859253
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4119453, 9.4076462
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8923187, 15.8875198
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3164558, 10.3125992
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7698288, 11.7634087
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2044983, 9.1955109
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7150269, 15.7131729
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4654541, 14.4571152
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7861938, 14.7799072
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6867485, 8.6887093
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3977890, 12.3969917
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8716850, 9.8716660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 518

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3437246, upper bound: 6.3257267
time: 11.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3401796, upper bound: 6.3292710
time: 20.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9671288, 11.9669342
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8085346, 6.8082943
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8891258, 6.8891544
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6144409, 9.6057053
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9575768, 8.9498405
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5017624, 11.4947662
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8378830, 9.8391647
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8987465, 8.8934975
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4541397, 9.4471416
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8814201, 10.8693352
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5815544, 11.5735512
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1944160, 8.1947632
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4140205, 11.4222946
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4748993, 14.4781075
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8624115, 16.8686142
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8532982, 9.8490143
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1135368, 10.1094589
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7111206, 13.7199097
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1297112, 11.1285896
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1954193, 9.1950531
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2519150, 12.2502747
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3841972, 8.3861828
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4932671, 9.5009155
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0350418, 12.0440636
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8704071, 11.8761063
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4218216, 10.4253807
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5155563, 11.5162201
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7205505, 6.7251854
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6870956, 12.6898651
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4019470, 11.4062424
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4038277, 9.4055710
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9040527, 15.9090042
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3194771, 10.3264694
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7796326, 11.7879562
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2198029, 9.2274895
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6918564, 15.7047272
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4858551, 14.4894257
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7945480, 14.8025894
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6834450, 8.6800194
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3799286, 12.3852539
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8667793, 9.8668900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3330628, upper bound: 6.3380928
time: 12.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3326166, upper bound: 6.3385390
time: 22.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9707718, 11.9672241
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8053665, 6.8038769
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8886719, 6.8879528
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6099091, 9.6025467
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9583778, 8.9509773
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4971390, 11.4918518
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8506775, 9.8566170
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8964691, 8.8928413
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4518394, 9.4486504
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8782578, 10.8673935
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5859833, 11.5765877
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1942482, 8.1947670
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4230347, 11.4294357
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4609451, 14.4595909
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8652115, 16.8704071
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8558617, 9.8513947
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1159363, 10.1120834
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7137527, 13.7234650
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1276283, 11.1287956
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1957207, 9.1954842
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2579727, 12.2558365
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3855438, 8.3860931
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4951897, 9.4996452
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0356598, 12.0399628
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8726616, 11.8768616
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4130859, 10.4190598
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5129318, 11.5152664
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7188740, 6.7221870
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6800003, 12.6837769
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4034348, 11.4044685
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4077339, 9.4089470
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9107208, 15.9118347
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3219452, 10.3245468
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7936935, 11.7963943
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2225037, 9.2278061
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6978989, 15.7043304
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4950943, 14.4962158
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7994385, 14.8008423
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6839371, 8.6827927
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3947067, 12.3984222
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8659477, 9.8671265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1415

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3267931, upper bound: 6.3422421
time: 21.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3259779, upper bound: 6.3430585
time: 23.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9713821, 11.9636536
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8074169, 6.8082714
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8886375, 6.8888168
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6149292, 9.6113625
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9582710, 8.9562302
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5030670, 11.4988251
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8390884, 9.8369980
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8985634, 8.8986282
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4468384, 9.4489784
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8840446, 10.8781853
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5848846, 11.5742912
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1969299, 8.1964607
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4200439, 11.4231186
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4706650, 14.4790382
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8623657, 16.8678436
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8541870, 9.8498650
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1164131, 10.1105537
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7116394, 13.7212906
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1319046, 11.1289673
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1953621, 9.1949387
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2499313, 12.2493896
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3861809, 8.3864517
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.5008926, 9.5027199
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0452499, 12.0456123
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8739738, 11.8775711
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4238358, 10.4266968
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5132370, 11.5169907
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7255974, 6.7257004
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6884460, 12.6905746
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4064140, 11.4035263
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4065666, 9.4042282
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9128952, 15.9101486
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3271942, 10.3235207
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7899094, 11.7899094
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2260818, 9.2310123
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7072678, 15.7074280
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4895782, 14.4934387
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8038635, 14.8061295
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6852875, 8.6782513
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3894043, 12.3864059
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8678093, 9.8657341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 657

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3373584, upper bound: 6.3361206
time: 15.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3310163, upper bound: 6.3438510
time: 13.82 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 31.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3456732, upper bound: 6.3263849
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3357463, upper bound: 6.3363112
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3464628, upper bound: 6.3362770
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3429104, upper bound: 6.3382432
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3350063, upper bound: 6.3323772
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3458655, upper bound: 6.3215194
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3444773, upper bound: 6.3274788
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3415469, upper bound: 6.3304107
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3462507, upper bound: 6.3285735
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3456147, upper bound: 6.3292118
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3437246, upper bound: 6.3257267
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3401796, upper bound: 6.3292710
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3330628, upper bound: 6.3380928
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3326166, upper bound: 6.3385390
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3267931, upper bound: 6.3422421
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3259779, upper bound: 6.3430585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3373584, upper bound: 6.3361206
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 31.56
Output dim: 26, lower bound: -6.3310163, upper bound: 6.3438510

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9620628, 11.9720764
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8074131, 6.8087044
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8887672, 6.8905792
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6126251, 9.6177559
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9540710, 8.9580650
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5001450, 11.5061836
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8355865, 9.8379593
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9012527, 8.9024353
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4549255, 9.4522419
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8785019, 10.8865929
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5766144, 11.5872726
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1993446, 8.1981201
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4291763, 11.4256363
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4799690, 14.4750252
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8718414, 16.8687897
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8515587, 9.8566704
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1075897, 10.1132622
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7231827, 13.7138214
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1290436, 11.1310539
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1961708, 9.1961441
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2502136, 12.2511139
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3861122, 8.3865280
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4969482, 9.4935493
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0390778, 12.0390472
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8752480, 11.8730850
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4211121, 10.4172249
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5174408, 11.5123520
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7251663, 6.7254162
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6920776, 12.6896515
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4005356, 11.4039268
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4022026, 9.4045181
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9095917, 15.9110222
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3259315, 10.3270035
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7908516, 11.7905197
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2244453, 9.2209873
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7075653, 15.7040939
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4833450, 14.4823990
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8007736, 14.8007278
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6842766, 8.6878433
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3837128, 12.3841057
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8640594, 9.8648643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3455928, upper bound: 6.3256147
time: 15.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449011, upper bound: 6.3263038
time: 12.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9586182, 11.9659081
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8066978, 6.8068905
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8906937, 6.8904381
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6138802, 9.6169586
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9557152, 8.9566422
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5025253, 11.5063591
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8314857, 9.8345509
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.9015617, 8.9015160
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4527588, 9.4496727
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8815842, 10.8852158
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5803986, 11.5859451
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1986427, 8.1980515
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4257278, 11.4246445
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4770546, 14.4736786
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8738022, 16.8702927
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8536301, 9.8566551
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1059685, 10.1098137
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7209625, 13.7154465
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1283035, 11.1306038
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1964874, 9.1963539
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2503052, 12.2507935
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3861198, 8.3865891
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4994354, 9.4967651
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0376129, 12.0371895
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8752174, 11.8737907
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4212723, 10.4200172
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5157013, 11.5116730
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7233429, 6.7237415
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6921082, 12.6902313
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4014893, 11.4045715
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4007454, 9.4031601
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9085464, 15.9107895
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3232079, 10.3248978
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7888489, 11.7886810
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2239037, 9.2224083
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7064056, 15.7053833
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4824142, 14.4836044
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8002701, 14.8002472
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6827278, 8.6867447
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3817062, 12.3834038
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8631058, 9.8647003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 934

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 659

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3446591, upper bound: 6.3344657
time: 22.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3446321, upper bound: 6.3344792
time: 13.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9516182, 11.9618950
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8019867, 6.8045635
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8819923, 6.8843193
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6015968, 9.6092606
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9412498, 8.9491463
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4901428, 11.4969330
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8351402, 9.8318844
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8885880, 8.8930321
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4382401, 9.4441662
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8595085, 10.8725548
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5582924, 11.5728149
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1998901, 8.1997871
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4218521, 11.4146156
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4880562, 14.4807968
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8717270, 16.8673096
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8437157, 9.8502312
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0864983, 10.0962143
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7180710, 13.7082291
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1258049, 11.1281052
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1960716, 9.1961632
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2502975, 12.2505798
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3788528, 8.3750172
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4943466, 9.4884453
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0396194, 12.0333405
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8656082, 11.8566933
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4157639, 10.4091072
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5137939, 11.5095215
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7215462, 6.7174225
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6854935, 12.6812134
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4007416, 11.3979073
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4061089, 9.4056129
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9067078, 15.9051132
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3293457, 10.3291931
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7866745, 11.7831154
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2121849, 9.2028408
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7061691, 15.7027969
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4729309, 14.4666748
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7950134, 14.7890701
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6691284, 8.6773109
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3893738, 12.3895798
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8675804, 9.8687668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3447630, upper bound: 6.3194387
time: 23.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3434601, upper bound: 6.3204103
time: 15.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9551086, 11.9603043
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7940121, 6.7980747
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8679161, 6.8727684
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.5933380, 9.6019859
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9325294, 8.9451180
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4836731, 11.4911423
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8211021, 9.8145294
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8671532, 8.8768539
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4121532, 9.4304581
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8430099, 10.8576164
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5561485, 11.5697212
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1939011, 8.1938820
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4455338, 11.4320831
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5036774, 14.4940910
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8674469, 16.8628235
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8384781, 9.8459702
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0749168, 10.0862122
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7262802, 13.7184296
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1145821, 11.1164551
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1952324, 9.1949768
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2467575, 12.2455101
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3650208, 8.3576298
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4902954, 9.4823875
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0284805, 12.0185623
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8551292, 11.8437347
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4229279, 10.4177742
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5265503, 11.5270157
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7116756, 6.7045498
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6780014, 12.6730576
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3821869, 11.3742409
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4092598, 9.4025764
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8801956, 15.8692398
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3059616, 10.2963791
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7573395, 11.7438202
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.1965981, 9.1813087
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6970596, 15.6868057
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4545746, 14.4425697
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7746735, 14.7584305
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6805038, 8.6877136
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3862381, 12.3812370
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8678970, 9.8684464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 626

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3398050, upper bound: 6.3193359
time: 11.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3363012, upper bound: 6.3228263
time: 12.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9527779, 11.9637184
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7935505, 6.7966766
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8712273, 6.8752956
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.5965080, 9.6046715
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9382057, 8.9431877
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4859543, 11.4927254
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8226280, 9.8182144
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8711014, 8.8756943
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4178066, 9.4260445
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8489838, 10.8586578
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5590668, 11.5739975
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1961403, 8.1969528
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4500999, 11.4418488
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.5009079, 14.4900589
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8687744, 16.8641815
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8421936, 9.8488045
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0798454, 10.0896263
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7238464, 13.7147522
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1147232, 11.1135597
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1952744, 9.1950722
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2475433, 12.2458839
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3658371, 8.3588009
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4851532, 9.4836922
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0214157, 12.0190239
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8543472, 11.8450089
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4221725, 10.4171486
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5290527, 11.5255775
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7090836, 6.7049313
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6798553, 12.6757889
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3785706, 11.3746338
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4054985, 9.4038048
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8752823, 15.8727188
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.2995720, 10.3019867
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7534027, 11.7495842
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.1959305, 9.1866608
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6911392, 15.6928940
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4557953, 14.4495049
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7719727, 14.7647095
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6798973, 8.6887016
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3793106, 12.3821640
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8645592, 9.8676567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 643

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 674

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449759, upper bound: 6.3282122
time: 11.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3457065, upper bound: 6.3267819
time: 32.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9547157, 11.9617805
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7941570, 6.7960663
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8718643, 6.8746605
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.5965881, 9.6045876
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9382057, 8.9431839
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4865112, 11.4921646
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8228760, 9.8179646
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8713799, 8.8754158
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4177990, 9.4260521
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8495483, 10.8580894
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5604248, 11.5726357
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1964417, 8.1966515
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4494438, 11.4425125
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4995575, 14.4914093
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8686523, 16.8643112
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8423309, 9.8486633
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0808983, 10.0885696
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7220993, 13.7164993
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1148834, 11.1133995
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1952972, 9.1950493
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2476501, 12.2457733
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3656235, 8.3590183
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4857101, 9.4831352
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0223923, 12.0180473
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8541946, 11.8451653
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4222641, 10.4170609
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5285034, 11.5261307
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7092285, 6.7047882
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6795807, 12.6760712
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3790512, 11.3741570
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4058533, 9.4034538
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8753586, 15.8726349
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3001671, 10.3013954
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7534943, 11.7494926
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.1953926, 9.1872025
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6912231, 15.6928024
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4547424, 14.4505501
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7720337, 14.7646408
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6799278, 8.6886711
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3801727, 12.3813057
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8650322, 9.8671875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 622

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 517

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3455389, upper bound: 6.3249991
time: 11.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3413980, upper bound: 6.3291359
time: 9.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9426651, 11.9494896
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.7933445, 6.7968082
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8704529, 6.8747730
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.5987053, 9.6037064
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9340096, 8.9398384
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4891968, 11.4926567
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8166351, 9.8125782
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8788261, 8.8835564
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4046631, 9.4158421
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8569489, 10.8660851
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5559692, 11.5641212
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1992302, 8.1979485
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4409790, 11.4321671
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4978638, 14.4918747
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8668060, 16.8630371
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8390656, 9.8437843
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0790443, 10.0859833
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7243805, 13.7180710
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1220436, 11.1199608
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1949806, 9.1949654
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2450943, 12.2434540
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3651810, 8.3599968
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4955482, 9.4961967
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0322571, 12.0296249
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8559113, 11.8496437
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4215698, 10.4194450
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5169601, 11.5169792
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7137508, 6.7097416
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6783066, 12.6753311
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.3920631, 11.3864212
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4098167, 9.4060516
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.8910828, 15.8865967
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3129158, 10.3099289
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7676430, 11.7615433
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2041893, 9.1952591
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7113037, 15.7103577
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4655304, 14.4572449
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7856140, 14.7794724
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6851349, 8.6874924
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3939667, 12.3941078
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8698616, 9.8702888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 888

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3353202, upper bound: 6.3179942
time: 19.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3353202, upper bound: 6.3179942
time: 19.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9709358, 11.9627151
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8070564, 6.8087959
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8880005, 6.8897324
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6113281, 9.6034317
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9562035, 8.9529114
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4982986, 11.4906425
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8326836, 9.8261051
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8970184, 8.8961830
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4466972, 9.4488544
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8818932, 10.8742065
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5787582, 11.5634308
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1976738, 8.1961479
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4198532, 11.4231949
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4703445, 14.4796143
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8561630, 16.8639832
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8517113, 9.8455048
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1160851, 10.1097908
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7057419, 13.7176208
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1294937, 11.1256180
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1925697, 9.1931458
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2498627, 12.2493019
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3838997, 8.3849220
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4896927, 9.4968948
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0362320, 12.0409775
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8680229, 11.8745079
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4189148, 10.4251060
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5084457, 11.5145340
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7226315, 6.7238884
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6857834, 12.6887741
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4064140, 11.4035149
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4035301, 9.3985138
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9124451, 15.9095459
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3239479, 10.3171997
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7883873, 11.7882767
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2256355, 9.2314777
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7010727, 15.7044678
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4816513, 14.4869499
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.8025131, 14.8065033
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6786995, 8.6659966
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3893280, 12.3866158
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8695488, 9.8648720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 525

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 514

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3308240, upper bound: 6.3436295
time: 15.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3307957, upper bound: 6.3436579
time: 13.20 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 31.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3455928, upper bound: 6.3256147
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3449011, upper bound: 6.3263038
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3446591, upper bound: 6.3344657
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3446321, upper bound: 6.3344792
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3447630, upper bound: 6.3194387
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3434601, upper bound: 6.3204103
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3398050, upper bound: 6.3193359
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3363012, upper bound: 6.3228263
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3449759, upper bound: 6.3282122
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3457065, upper bound: 6.3267819
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3455389, upper bound: 6.3249991
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3413980, upper bound: 6.3291359
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3353202, upper bound: 6.3179942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3353202, upper bound: 6.3179942
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3308240, upper bound: 6.3436295
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 31.02
Output dim: 26, lower bound: -6.3307957, upper bound: 6.3436579

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9615669, 11.9725533
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8081512, 6.8093014
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8882217, 6.8899689
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6111298, 9.6161804
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9514580, 8.9544144
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4996262, 11.5055428
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8269272, 9.8314304
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8988380, 8.8992310
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4526386, 9.4493847
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8781128, 10.8863831
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5747643, 11.5858078
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1992645, 8.1973953
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4260025, 11.4232407
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4746552, 14.4684906
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8695755, 16.8659134
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8517685, 9.8570595
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1096001, 10.1165390
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7210770, 13.7111359
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1279831, 11.1303139
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1952324, 9.1949043
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2488098, 12.2490730
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3854866, 8.3855419
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4970779, 9.4937401
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0390701, 12.0390129
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8740997, 11.8715630
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4209595, 10.4171066
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5164185, 11.5110474
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7248344, 6.7250748
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6917419, 12.6888657
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4013748, 11.4048309
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.3969994, 9.4006004
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9019470, 15.9052620
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3200188, 10.3225403
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7896042, 11.7893639
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2249756, 9.2213631
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6982269, 15.6970444
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4861526, 14.4844398
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7992783, 14.7993546
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6719856, 8.6785660
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3759384, 12.3782349
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8556099, 9.8584938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 947

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 643

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443013, upper bound: 6.3234232
time: 9.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3433611, upper bound: 6.3243852
time: 22.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9625397, 11.9715767
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8080101, 6.8094425
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8881569, 6.8900337
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6110497, 9.6162605
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9504204, 8.9554520
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4994965, 11.5056648
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8290558, 9.8293018
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8980522, 8.9000168
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4520664, 9.4499569
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8782959, 10.8862038
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5751534, 11.5854263
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1986198, 8.1980438
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4267807, 11.4224701
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4734344, 14.4697113
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8689651, 16.8665237
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8519516, 9.8568802
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1108627, 10.1152725
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7204971, 13.7117157
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1283035, 11.1299934
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1949310, 9.1952095
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2481689, 12.2497215
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3851280, 8.3859043
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4971390, 9.4936790
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0390396, 12.0390472
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8737259, 11.8719368
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4210052, 10.4170609
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5161438, 11.5113258
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7248230, 6.7250862
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6912918, 12.6893158
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4014359, 11.4047661
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.3982849, 9.3993149
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9038239, 15.9033775
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3214722, 10.3210869
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7896957, 11.7892799
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2248192, 9.2215195
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7005157, 15.6947479
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4853897, 14.4852028
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7994003, 14.7992325
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6750031, 8.6755505
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3778458, 12.3763275
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8576889, 9.8564148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 824

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 722

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3319622, upper bound: 6.3261023
time: 11.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3446996, upper bound: 6.3133650
time: 12.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9527168, 11.9625473
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8061161, 6.8062954
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8897743, 6.8898220
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6071625, 9.6148605
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9482803, 8.9544716
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4948120, 11.5040092
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8309441, 9.8320026
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8973694, 8.8989601
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4512215, 9.4492092
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8714676, 10.8822174
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5677528, 11.5821953
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1986122, 8.1980133
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4237366, 11.4164505
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4816589, 14.4715958
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8780899, 16.8691101
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8474236, 9.8548164
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0992889, 10.1060944
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7170792, 13.7025833
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1249619, 11.1298447
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1964493, 9.1969147
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2500458, 12.2519341
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3855705, 8.3860664
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4958801, 9.4902496
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0365143, 12.0334816
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8728218, 11.8684044
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4219017, 10.4181099
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5142593, 11.5090141
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7228413, 6.7223892
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6927719, 12.6891632
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4013748, 11.4061127
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4007225, 9.4030418
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9071808, 15.9061966
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3244896, 10.3247070
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7873039, 11.7824707
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2225838, 9.2143669
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7030945, 15.6949081
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4800949, 14.4768295
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7977753, 14.7911148
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6804810, 8.6820831
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3809967, 12.3807030
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8630753, 9.8644714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1017

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3442038, upper bound: 6.3243828
time: 25.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3345721, upper bound: 6.3340086
time: 12.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9551849, 11.9600067
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8061047, 6.8063068
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8900757, 6.8895206
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.6116333, 9.6102409
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9535484, 8.9492073
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.5001144, 11.4986458
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8289375, 9.8340111
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8987541, 8.8973236
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4522820, 9.4481354
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8785629, 10.8750992
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5766487, 11.5733032
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1985970, 8.1980209
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4175339, 11.4226570
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4749756, 14.4782791
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8726120, 16.8745880
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8517876, 9.8504524
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.1022034, 10.1031342
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7080994, 13.7115555
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1275406, 11.1272697
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1970444, 9.1963196
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2514420, 12.2505341
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3855972, 8.3860359
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4929199, 9.4927483
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0339050, 12.0359840
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8698311, 11.8713913
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4193687, 10.4205055
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5130463, 11.5101852
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7219944, 6.7232361
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6910400, 12.6908875
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4028931, 11.4044609
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4006271, 9.4031410
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9039536, 15.9094238
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3230171, 10.3261795
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7826347, 11.7871399
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2158661, 9.2210846
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.6959229, 15.7020340
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4756393, 14.4812851
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7911377, 14.7977524
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6777153, 8.6844978
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3789978, 12.3826904
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8628769, 9.8646698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 607

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1686

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3418059, upper bound: 6.3335283
time: 13.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3436812, upper bound: 6.3316592
time: 13.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -26.9983959, -9.4878750, -26.9983959, -9.4878750, -11.9516182, 11.9612427
1: -9.8015137, -0.0086074, -9.8015137, -0.0086074, -6.8029766, 6.8042526
2: -4.3753042, 4.8853683, -4.3753042, 4.8853683, -6.8829956, 6.8839188
3: -13.5000191, -0.6282945, -13.5000191, -0.6282945, -9.5997047, 9.6091118
4: -5.5107589, 7.3384104, -5.5107589, 7.3384104, -8.9404259, 8.9485893
5: -8.9504814, 4.1939621, -8.9504814, 4.1939621, -11.4871902, 11.4963226
6: -24.1284771, -8.9163694, -24.1284771, -8.9163694, -9.8279419, 9.8272324
7: -9.7015333, 2.6900015, -9.7015333, 2.6900015, -8.8878937, 8.8927002
8: -12.3511038, 3.1013632, -12.3511038, 3.1013632, -9.4375496, 9.4438457
9: -7.1024103, 8.6224747, -7.1024103, 8.6224747, -10.8592911, 10.8724289
10: -7.0787110, 7.1774583, -7.0787110, 7.1774583, -11.5553131, 11.5720825
11: -4.6735649, 5.0700879, -4.6735649, 5.0700879, -8.1993408, 8.2002602
12: -16.8777847, -0.5807475, -16.8777847, -0.5807475, -11.4217148, 11.4144173
13: -21.3878517, -3.0878239, -21.3878517, -3.0878239, -14.4877319, 14.4776344
14: -22.9096756, -5.0321493, -22.9096756, -5.0321493, -16.8704758, 16.8643951
15: -9.0647802, 3.4752245, -9.0647802, 3.4752245, -9.8426094, 9.8498344
16: -9.6627026, 1.1415594, -9.6627026, 1.1415594, -10.0859795, 10.0960007
17: -20.8171902, -4.1509328, -20.8171902, -4.1509328, -13.7173004, 13.7067261
18: -3.2264037, 11.8097839, -3.2264037, 11.8097839, -11.1219368, 11.1265297
19: 1.8459659, 11.0984392, 1.8459659, 11.0984392, -9.1957703, 9.1957436
20: -0.8079145, 9.8933392, -0.8079145, 9.8933392, -10.7012539, 10.7012539
21: 0.7482438, 13.1297464, 0.7482438, 13.1297464, -12.2501984, 12.2505913
22: 1.9828215, 12.3014107, 1.9828215, 12.3014107, -8.3783989, 8.3740177
23: 0.2048931, 11.1095238, 0.2048931, 11.1095238, -9.4923401, 9.4824409
24: -5.3671875, 9.5296078, -5.3671875, 9.5296078, -12.0386200, 12.0296211
25: -4.3910112, 9.7639790, -4.3910112, 9.7639790, -11.8647308, 11.8541031
26: 2.9777021, 16.3151169, 2.9777021, 16.3151169, -13.3374147, 13.3374147
27: 0.0914311, 12.2316818, 0.0914311, 12.2316818, -10.4157295, 10.4083900
28: 0.7322710, 12.6663208, 0.7322710, 12.6663208, -11.5133591, 11.5082779
29: -0.4704075, 9.0134182, -0.4704075, 9.0134182, -6.7211475, 6.7166214
30: -4.0190020, 9.9890909, -4.0190020, 9.9890909, -12.6849442, 12.6802979
31: -3.0747058, 11.8268099, -3.0747058, 11.8268099, -11.4002838, 11.3972015
32: -19.0569973, -5.8382254, -19.0569973, -5.8382254, -9.4040146, 9.4036789
33: -38.4278183, -16.8330193, -38.4278183, -16.8330193, -15.9050293, 15.9023132
34: -37.8674545, -23.4178619, -37.8674545, -23.4178619, -10.3291702, 10.3290253
35: -29.0156307, -14.0512800, -29.0156307, -14.0512800, -11.7859116, 11.7799911
36: -22.0339241, -9.1940041, -22.0339241, -9.1940041, -9.2119827, 9.2015438
37: -39.6359177, -18.9760094, -39.6359177, -18.9760094, -15.7064438, 15.6998596
38: -36.0299835, -19.3349819, -36.0299835, -19.3349819, -14.4718552, 14.4661636
39: -38.3859787, -16.9133701, -38.3859787, -16.9133701, -14.7962189, 14.7859573
40: -34.3942947, -20.4554405, -34.3942947, -20.4554405, -8.6620178, 8.6749001
41: -21.2123260, -5.2898879, -21.2123260, -5.2898879, -12.3897552, 12.3894882
42: -23.4739628, -11.3906031, -23.4739628, -11.3906031, -9.8674622, 9.8702011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=65, inp2_unstable=65, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 640

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 702

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3193906, upper bound: 6.3183934
time: 22.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3437193, upper bound: 6.2940665
time: 12.20 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 36.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3443013, upper bound: 6.3234232
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3433611, upper bound: 6.3243852
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3319622, upper bound: 6.3261023
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3446996, upper bound: 6.3133650
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3442038, upper bound: 6.3243828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3345721, upper bound: 6.3340086
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3418059, upper bound: 6.3335283
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3436812, upper bound: 6.3316592
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3193906, upper bound: 6.3183934
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 36.58
Output dim: 26, lower bound: -6.3437193, upper bound: 6.2940665
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 36.58
Output dim: 26, lower bound: -6.3449759, upper bound: 6.3282122
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 36.58
Output dim: 26, lower bound: -6.3457065, upper bound: 6.3267819
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 36.58
Output dim: 26, lower bound: -6.3455389, upper bound: 6.3249991
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 36.58
Output dim: 26, lower bound: -6.3308240, upper bound: 6.3436295
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 36.58
Output dim: 26, lower bound: -6.3307957, upper bound: 6.3436579

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 22.82 + 1796.75 = 1819.57 seconds
