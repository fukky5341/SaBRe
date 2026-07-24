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
execution time: IAR + RelationalAnalysis = 2.71 + 20.75 = 23.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 26, lower bound: -6.3498704, upper bound: 6.3498704

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 657

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3496383, upper bound: 6.3422455
time: 12.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3496383, upper bound: 6.3496382
time: 11.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 24.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 24.64
Output dim: 26, lower bound: -6.3496383, upper bound: 6.3422455
IS_A2, status: Status.UNKNOWN, split count: 1, time: 24.64
Output dim: 26, lower bound: -6.3496383, upper bound: 6.3496382

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -26.9882011, -9.4928255, -26.9950256, -9.4895630, -11.9725685, 11.9760246
1: -9.7958965, -0.0114202, -9.7996187, -0.0095444, -6.8002338, 6.8048325
2: -4.3729668, 4.8838711, -4.3745341, 4.8848553, -6.8806725, 6.8837013
3: -13.4855804, -0.6298409, -13.4952021, -0.6288028, -9.6097260, 9.6172066
4: -5.4889746, 7.3368936, -5.5034809, 7.3379107, -8.9413033, 8.9556007
5: -8.9339466, 4.1929145, -8.9449663, 4.1935954, -11.4873199, 11.4982147
6: -24.1022987, -8.9176188, -24.1197262, -8.9167948, -9.8204803, 9.8371201
7: -9.6935492, 2.6891317, -9.6987524, 2.6896992, -8.8960991, 8.9011040
8: -12.3454199, 3.0996709, -12.3491592, 3.1008143, -9.4495945, 9.4522572
9: -7.0869513, 8.6199703, -7.0972672, 8.6216259, -10.8783951, 10.8863144
10: -7.0533628, 7.1736636, -7.0703039, 7.1761780, -11.5695419, 11.5835228
11: -4.6712503, 5.0599165, -4.6727810, 5.0666938, -8.1930504, 8.1853333
12: -16.8745823, -0.5851787, -16.8766956, -0.5822599, -11.4221458, 11.4238586
13: -21.3857327, -3.0947485, -21.3871231, -3.0901260, -14.4759216, 14.4763680
14: -22.9044666, -5.0529032, -22.9079475, -5.0390778, -16.8620453, 16.8522263
15: -9.0500479, 3.4709566, -9.0599012, 3.4737706, -9.8455544, 9.8520889
16: -9.6494322, 1.1392422, -9.6582832, 1.1407828, -10.1007309, 10.1076736
17: -20.8131142, -4.1712022, -20.8158073, -4.1576986, -13.7234879, 13.7119293
18: -3.2208588, 11.8082256, -3.2245364, 11.8092632, -11.1255646, 11.1306458
19: 1.8492208, 11.0846224, 1.8470564, 11.0938072, -9.1894798, 9.1822205
20: -0.8023248, 9.8857231, -0.8060265, 9.8907690, -10.6930943, 10.6917496
21: 0.7530012, 13.1168022, 0.7498314, 13.1252575, -12.2422409, 12.2352486
22: 1.9851294, 12.2912703, 1.9836040, 12.2979231, -8.3825302, 8.3764820
23: 0.2076104, 11.0746861, 0.2058053, 11.0979519, -9.4948997, 9.4732742
24: -5.3650551, 9.5055513, -5.3664727, 9.5216274, -12.0447922, 12.0301323
25: -4.3883219, 9.7337379, -4.3901052, 9.7539082, -11.8685226, 11.8497429
26: 2.9832211, 16.3013382, 2.9795399, 16.3104858, -13.3272648, 13.3217983
27: 0.0945470, 12.2186956, 0.0924528, 12.2273207, -10.4203644, 10.4168358
28: 0.7361376, 12.6393642, 0.7335665, 12.6573782, -11.5089035, 11.4931488
29: -0.4691657, 9.0014811, -0.4699920, 9.0094032, -6.7248535, 6.7172756
30: -4.0155878, 9.9684334, -4.0178537, 9.9822426, -12.6814957, 12.6691360
31: -3.0707235, 11.8162518, -3.0733724, 11.8232212, -11.4031448, 11.3980942
32: -19.0379963, -5.8404293, -19.0506287, -5.8389616, -9.3918304, 9.4033394
33: -38.4239044, -16.8392658, -38.4264984, -16.8351040, -15.9123230, 15.9096985
34: -37.8543167, -23.4214993, -37.8630257, -23.4191074, -10.3226204, 10.3293304
35: -29.0127029, -14.0571842, -29.0146389, -14.0533161, -11.7924423, 11.7896690
36: -22.0319653, -9.2000179, -22.0332623, -9.1960125, -9.2240295, 9.2232819
37: -39.6327515, -18.9874363, -39.6348343, -18.9798717, -15.7123108, 15.7073135
38: -36.0212669, -19.3388710, -36.0269203, -19.3362923, -14.4742355, 14.4797897
39: -38.3828735, -16.9172134, -38.3849030, -16.9146481, -14.8002167, 14.8039246
40: -34.3684464, -20.4562721, -34.3856125, -20.4557247, -8.6666641, 8.6840534
41: -21.2008743, -5.2938833, -21.2084789, -5.2912741, -12.3789215, 12.3855972
42: -23.4718933, -11.3937254, -23.4732361, -11.3916483, -9.8655968, 9.8615456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=64, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 627

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3481806, upper bound: 6.3270271
time: 13.69 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3481806, upper bound: 6.3408075
time: 27.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -26.9978123, -9.4771175, -26.9979668, -9.4881945, -11.9819489, 11.9932327
1: -9.8012438, 0.0004890, -9.8012915, -0.0087585, -6.8062954, 6.8235893
2: -4.3762999, 4.8880458, -4.3752308, 4.8852901, -6.8842888, 6.8964481
3: -13.4992170, -0.6116352, -13.4991837, -0.6283522, -9.6209221, 9.6394424
4: -5.5110312, 7.3551149, -5.5102634, 7.3383503, -8.9631271, 8.9814568
5: -8.9504652, 4.2090940, -8.9496193, 4.1938949, -11.5021210, 11.5219154
6: -24.1285362, -8.8987713, -24.1279793, -8.9164858, -9.8405914, 9.8643036
7: -9.7016001, 2.6994133, -9.7009258, 2.6899581, -8.9043121, 8.9149170
8: -12.3509045, 3.1049044, -12.3503284, 3.1013105, -9.4575691, 9.4607563
9: -7.1015296, 8.6359596, -7.1015644, 8.6223278, -10.8922882, 10.9074783
10: -7.0788093, 7.2009010, -7.0774169, 7.1772537, -11.5918922, 11.6207237
11: -4.6820602, 5.0706477, -4.6734648, 5.0696249, -8.2116966, 8.1978188
12: -16.8806591, -0.5794702, -16.8771076, -0.5810108, -11.4333344, 11.4371948
13: -21.3893089, -3.0855422, -21.3872871, -3.0880504, -14.4824982, 14.4932632
14: -22.9260902, -5.0323029, -22.9094124, -5.0327244, -16.8904495, 16.8713989
15: -9.0651455, 3.4875627, -9.0639153, 3.4750693, -9.8607559, 9.8741379
16: -9.6617651, 1.1541419, -9.6621151, 1.1414032, -10.1125336, 10.1256561
17: -20.8315372, -4.1519918, -20.8169746, -4.1517792, -13.7511444, 13.7320862
18: -3.2307463, 11.8142967, -3.2262075, 11.8096514, -11.1354713, 11.1397781
19: 1.8338568, 11.0977983, 1.8460686, 11.0979462, -9.2100143, 9.1949768
20: -0.8186200, 9.8929319, -0.8076823, 9.8932028, -10.7118225, 10.7006140
21: 0.7362430, 13.1294489, 0.7483990, 13.1290493, -12.2642059, 12.2502518
22: 1.9701982, 12.3015566, 1.9829159, 12.3010807, -8.4015350, 8.3867569
23: 0.1768827, 11.1081333, 0.2049890, 11.1088762, -9.5363388, 9.5021782
24: -5.3884873, 9.5290956, -5.3671045, 9.5291328, -12.0756378, 12.0501976
25: -4.4159288, 9.7645130, -4.3908706, 9.7634134, -11.9061852, 11.8790283
26: 2.9561925, 16.3148918, 2.9779501, 16.3148212, -13.3586292, 13.3369417
27: 0.0773606, 12.2313595, 0.0915821, 12.2314720, -10.4378586, 10.4254570
28: 0.7099228, 12.6662388, 0.7324541, 12.6658344, -11.5434952, 11.5190887
29: -0.4820260, 9.0147133, -0.4703379, 9.0131588, -6.7418785, 6.7297153
30: -4.0365410, 9.9893084, -4.0188632, 9.9886894, -12.7103577, 12.6915817
31: -3.0840650, 11.8278112, -3.0745904, 11.8261690, -11.4199867, 11.4110832
32: -19.0586758, -5.8253374, -19.0565662, -5.8383980, -9.4097176, 9.4238815
33: -38.4302711, -16.8322639, -38.4276428, -16.8338432, -15.9219894, 15.9183884
34: -37.8685150, -23.4097443, -37.8670883, -23.4180622, -10.3350716, 10.3459816
35: -29.0159187, -14.0509272, -29.0154839, -14.0523186, -11.7973328, 11.7973137
36: -22.0380898, -9.1941471, -22.0338058, -9.1944523, -9.2299652, 9.2319641
37: -39.6474609, -18.9767494, -39.6357841, -18.9770985, -15.7275925, 15.7181625
38: -36.0312538, -19.3289871, -36.0292435, -19.3351631, -14.4831390, 14.4986305
39: -38.3875771, -16.9131184, -38.3858109, -16.9139042, -14.8063736, 14.8160095
40: -34.3943214, -20.4381428, -34.3937340, -20.4554520, -8.6874123, 8.7111282
41: -21.2130051, -5.2824469, -21.2120590, -5.2901149, -12.3920288, 12.4035110
42: -23.4726887, -11.3912125, -23.4727192, -11.3907633, -9.8779488, 9.8647194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=64, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 627

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3481806, upper bound: 6.3344208
time: 14.28 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3481806, upper bound: 6.3481801
time: 15.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 32.28 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 32.28
Output dim: 26, lower bound: -6.3481806, upper bound: 6.3270271
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 32.28
Output dim: 26, lower bound: -6.3481806, upper bound: 6.3408075
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 32.28
Output dim: 26, lower bound: -6.3481806, upper bound: 6.3344208
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 32.28
Output dim: 26, lower bound: -6.3481806, upper bound: 6.3481801

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -26.9770355, -9.4951296, -26.9950256, -9.4895630, -11.9615479, 11.9744453
1: -9.7891846, -0.0130415, -9.7996187, -0.0095444, -6.7935810, 6.8036842
2: -4.3690886, 4.8832159, -4.3745341, 4.8848553, -6.8752327, 6.8830299
3: -13.4704370, -0.6304593, -13.4952021, -0.6288028, -9.5946884, 9.6166077
4: -5.4728041, 7.3361382, -5.5034809, 7.3379107, -8.9256058, 8.9543533
5: -8.9192896, 4.1921697, -8.9449663, 4.1935954, -11.4727631, 11.4975052
6: -24.1012802, -8.9200764, -24.1197262, -8.9167948, -9.8190498, 9.8347073
7: -9.6866617, 2.6886206, -9.6987524, 2.6896992, -8.8893318, 8.9004211
8: -12.3358755, 3.0991976, -12.3491592, 3.1008143, -9.4411945, 9.4518814
9: -7.0647936, 8.6190205, -7.0972672, 8.6216259, -10.8562660, 10.8854713
10: -7.0281510, 7.1717572, -7.0703039, 7.1761780, -11.5444908, 11.5818405
11: -4.6699343, 5.0446329, -4.6727810, 5.0666938, -8.1919403, 8.1699409
12: -16.8740673, -0.5903882, -16.8766956, -0.5822599, -11.4203148, 11.4197731
13: -21.3737564, -3.0973477, -21.3871231, -3.0901260, -14.4632187, 14.4746971
14: -22.9026833, -5.0549841, -22.9079475, -5.0390778, -16.8566742, 16.8500519
15: -9.0324545, 3.4695497, -9.0599012, 3.4737706, -9.8281212, 9.8504066
16: -9.6351738, 1.1375327, -9.6582832, 1.1407828, -10.0867691, 10.1065445
17: -20.8105545, -4.1773529, -20.8158073, -4.1576986, -13.7194824, 13.7080460
18: -3.2195089, 11.8027763, -3.2245364, 11.8092632, -11.1244812, 11.1225777
19: 1.8512347, 11.0681458, 1.8470564, 11.0938072, -9.1880836, 9.1657028
20: -0.7985914, 9.8744850, -0.8060265, 9.8907690, -10.6893606, 10.6805115
21: 0.7562511, 13.0984726, 0.7498314, 13.1252575, -12.2401123, 12.2160835
22: 1.9868393, 12.2825155, 1.9836040, 12.2979231, -8.3812294, 8.3669300
23: 0.2092534, 11.0504189, 0.2058053, 11.0979519, -9.4934425, 9.4488716
24: -5.3634548, 9.4828968, -5.3664727, 9.5216274, -12.0434265, 12.0074348
25: -4.3862185, 9.7126093, -4.3901052, 9.7539082, -11.8662949, 11.8285980
26: 2.9872212, 16.2884865, 2.9795399, 16.3104858, -13.3232651, 13.3089466
27: 0.0956895, 12.2023544, 0.0924528, 12.2273207, -10.4193268, 10.4008560
28: 0.7393351, 12.6200962, 0.7335665, 12.6573782, -11.5063248, 11.4738846
29: -0.4684353, 8.9915886, -0.4699920, 9.0094032, -6.7241135, 6.7073689
30: -4.0136514, 9.9512539, -4.0178537, 9.9822426, -12.6794128, 12.6522827
31: -3.0686016, 11.7996569, -3.0733724, 11.8232212, -11.4011993, 11.3813782
32: -19.0364666, -5.8427048, -19.0506287, -5.8389616, -9.3861542, 9.4015350
33: -38.4216309, -16.8494606, -38.4264984, -16.8351040, -15.9108734, 15.8974609
34: -37.8527908, -23.4245148, -37.8630257, -23.4191074, -10.3198547, 10.3262672
35: -29.0110340, -14.0665417, -29.0146389, -14.0533161, -11.7913055, 11.7798691
36: -22.0307846, -9.2087822, -22.0332623, -9.1960125, -9.2233925, 9.2150421
37: -39.6312943, -19.0036163, -39.6348343, -18.9798717, -15.7112045, 15.6912155
38: -36.0196762, -19.3460579, -36.0269203, -19.3362923, -14.4733734, 14.4717331
39: -38.3805084, -16.9230194, -38.3849030, -16.9146481, -14.7984085, 14.7991638
40: -34.3644867, -20.4572430, -34.3856125, -20.4557247, -8.6624146, 8.6829929
41: -21.2001591, -5.2984762, -21.2084789, -5.2912741, -12.3776398, 12.3801880
42: -23.4708118, -11.3958893, -23.4732361, -11.3916483, -9.8645287, 9.8586807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=63, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3344209, upper bound: 6.3270271
time: 11.62 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3344209, upper bound: 6.3270271
time: 8.58 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -26.9879570, -9.4532700, -26.9944630, -9.4898214, -11.9721184, 12.0165977
1: -9.7954330, 0.0124958, -9.7992477, -0.0097637, -6.7997189, 6.8295155
2: -4.3744130, 4.8961697, -4.3742599, 4.8847275, -6.8781853, 6.8995056
3: -13.4853401, -0.5861940, -13.4942188, -0.6289229, -9.6097221, 9.6602058
4: -5.4897037, 7.3780522, -5.5025964, 7.3377142, -8.9428177, 8.9956512
5: -8.9346542, 4.2350836, -8.9440956, 4.1934891, -11.4880066, 11.5396957
6: -24.0912685, -8.9204521, -24.1146755, -8.9171591, -9.8147545, 9.8341236
7: -9.6942177, 2.7100329, -9.6980867, 2.6896586, -8.8971443, 8.9215088
8: -12.3466339, 3.1207530, -12.3485394, 3.1006508, -9.4518108, 9.4720821
9: -7.0854883, 8.6840382, -7.0959854, 8.6215019, -10.8775902, 10.9491730
10: -7.0520515, 7.2423353, -7.0688801, 7.1759071, -11.5689163, 11.6517754
11: -4.7134833, 5.0607681, -4.6725187, 5.0658250, -8.2348213, 8.1852226
12: -16.8726101, -0.5827066, -16.8748550, -0.5827880, -11.4211807, 11.4434624
13: -21.3870201, -3.0575919, -21.3864498, -3.0904131, -14.4755707, 14.5149460
14: -22.9094810, -5.0493355, -22.9075203, -5.0392647, -16.8621292, 16.8654861
15: -9.0498352, 3.5126982, -9.0588741, 3.4735146, -9.8460426, 9.8926811
16: -9.6480761, 1.1805964, -9.6572847, 1.1406112, -10.0994949, 10.1495438
17: -20.8266201, -4.1689920, -20.8154621, -4.1581440, -13.7301178, 13.7181473
18: -3.2414110, 11.8031712, -3.2242830, 11.8063860, -11.1525536, 11.1300430
19: 1.8016522, 11.0828094, 1.8472247, 11.0929146, -9.2367096, 9.1782684
20: -0.8375182, 9.8848057, -0.8057003, 9.8902092, -10.7277279, 10.6905060
21: 0.6987576, 13.1158466, 0.7501647, 13.1242523, -12.2972870, 12.2326355
22: 1.9568987, 12.2897081, 1.9837809, 12.2968578, -8.4128036, 8.3752670
23: 0.1340023, 11.0731163, 0.2060239, 11.0967102, -9.5675964, 9.4708252
24: -5.4303989, 9.5038891, -5.3662057, 9.5203362, -12.1089630, 12.0286674
25: -4.4500484, 9.7330084, -4.3898730, 9.7527256, -11.9290962, 11.8490143
26: 2.9352660, 16.2997856, 2.9799399, 16.3081169, -13.3728504, 13.3198452
27: 0.0496600, 12.2169590, 0.0927429, 12.2264376, -10.4651031, 10.4159737
28: 0.6806746, 12.6392546, 0.7338452, 12.6563749, -11.5643158, 11.4924278
29: -0.4985011, 9.0025635, -0.4698945, 9.0087805, -6.7536640, 6.7183056
30: -4.0631890, 9.9685135, -4.0175586, 9.9813166, -12.7284698, 12.6697311
31: -3.1251707, 11.8153009, -3.0732050, 11.8222599, -11.4566956, 11.3961868
32: -19.0267887, -5.8402996, -19.0447807, -5.8392591, -9.3895645, 9.4202652
33: -38.4520340, -16.8387032, -38.4262161, -16.8357563, -15.9485092, 15.9057159
34: -37.8472786, -23.4226036, -37.8596649, -23.4193649, -10.3201408, 10.3308716
35: -29.0364456, -14.0568829, -29.0142441, -14.0539293, -11.8179398, 11.7887230
36: -22.0482349, -9.2005405, -22.0330410, -9.1965809, -9.2405701, 9.2235909
37: -39.6759338, -18.9866238, -39.6346130, -18.9808483, -15.7551346, 15.7080078
38: -36.0390930, -19.3379421, -36.0266609, -19.3369293, -14.4955978, 14.4780426
39: -38.4013824, -16.9181061, -38.3847504, -16.9151344, -14.8172836, 14.8025742
40: -34.3551788, -20.4479027, -34.3793411, -20.4558525, -8.6605339, 8.6969032
41: -21.2055168, -5.2923131, -21.2082100, -5.2918720, -12.3816071, 12.3879204
42: -23.4738426, -11.3910522, -23.4727268, -11.3919611, -9.8711891, 9.8624458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=63, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 659

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3463212, upper bound: 6.3397723
time: 28.36 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3463212, upper bound: 6.3389907
time: 20.57 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -26.9866142, -9.4793701, -26.9979668, -9.4881945, -11.9709244, 11.9916649
1: -9.7945299, -0.0011213, -9.8012915, -0.0087585, -6.7996349, 6.8224487
2: -4.3724108, 4.8874083, -4.3752308, 4.8852901, -6.8788528, 6.8957882
3: -13.4840717, -0.6122746, -13.4991837, -0.6283522, -9.6058846, 9.6388588
4: -5.4948850, 7.3543296, -5.5102634, 7.3383503, -8.9474144, 8.9802132
5: -8.9358091, 4.2084088, -8.9496193, 4.1938949, -11.4875641, 11.5211945
6: -24.1275177, -8.9011860, -24.1279793, -8.9164858, -9.8391609, 9.8618851
7: -9.6947165, 2.6988602, -9.7009258, 2.6899581, -8.8975296, 8.9142494
8: -12.3413506, 3.1044726, -12.3503284, 3.1013105, -9.4491615, 9.4603825
9: -7.0794077, 8.6349897, -7.1015644, 8.6223278, -10.8701744, 10.9066467
10: -7.0536256, 7.1989841, -7.0774169, 7.1772537, -11.5668373, 11.6190567
11: -4.6807418, 5.0553465, -4.6734648, 5.0696249, -8.2106094, 8.1824265
12: -16.8801346, -0.5846779, -16.8771076, -0.5810108, -11.4315033, 11.4331017
13: -21.3773556, -3.0881143, -21.3872871, -3.0880504, -14.4698105, 14.4915924
14: -22.9243336, -5.0344229, -22.9094124, -5.0327244, -16.8850784, 16.8691864
15: -9.0475464, 3.4861650, -9.0639153, 3.4750693, -9.8433075, 9.8724556
16: -9.6474676, 1.1524248, -9.6621151, 1.1414032, -10.0985374, 10.1245193
17: -20.8289566, -4.1581349, -20.8169746, -4.1517792, -13.7471466, 13.7282181
18: -3.2294054, 11.8088322, -3.2262075, 11.8096514, -11.1343880, 11.1317101
19: 1.8358788, 11.0813417, 1.8460686, 11.0979462, -9.2086143, 9.1784477
20: -0.8148978, 9.8817072, -0.8076823, 9.8932028, -10.7081003, 10.6893892
21: 0.7394619, 13.1111174, 0.7483990, 13.1290493, -12.2621002, 12.2310944
22: 1.9718971, 12.2928066, 1.9829159, 12.3010807, -8.4002304, 8.3772011
23: 0.1785188, 11.0838699, 0.2049890, 11.1088762, -9.5348892, 9.4777641
24: -5.3868694, 9.5064573, -5.3671045, 9.5291328, -12.0742493, 12.0275002
25: -4.4138260, 9.7433720, -4.3908706, 9.7634134, -11.9039803, 11.8578835
26: 2.9601517, 16.3020153, 2.9779501, 16.3148212, -13.3546696, 13.3240652
27: 0.0785117, 12.2150259, 0.0915821, 12.2314720, -10.4368134, 10.4094582
28: 0.7131112, 12.6469707, 0.7324541, 12.6658344, -11.5409164, 11.4998093
29: -0.4813317, 9.0048370, -0.4703379, 9.0131588, -6.7411537, 6.7198181
30: -4.0345783, 9.9721451, -4.0188632, 9.9886894, -12.7083130, 12.6747208
31: -3.0819690, 11.8112020, -3.0745904, 11.8261690, -11.4180336, 11.3943481
32: -19.0571251, -5.8276176, -19.0565662, -5.8383980, -9.4040527, 9.4220924
33: -38.4280205, -16.8424454, -38.4276428, -16.8338432, -15.9205475, 15.9061661
34: -37.8670311, -23.4127884, -37.8670883, -23.4180622, -10.3323021, 10.3429184
35: -29.0142422, -14.0602970, -29.0154839, -14.0523186, -11.7961960, 11.7875214
36: -22.0369511, -9.2028999, -22.0338058, -9.1944523, -9.2293282, 9.2237320
37: -39.6460190, -18.9929390, -39.6357841, -18.9770985, -15.7264786, 15.7020645
38: -36.0296936, -19.3361969, -36.0292435, -19.3351631, -14.4822540, 14.4906044
39: -38.3852158, -16.9188824, -38.3858109, -16.9139042, -14.8045654, 14.8112564
40: -34.3903427, -20.4390736, -34.3937340, -20.4554520, -8.6831741, 8.7100677
41: -21.2122765, -5.2870321, -21.2120590, -5.2901149, -12.3907166, 12.3981247
42: -23.4716148, -11.3933945, -23.4727192, -11.3907633, -9.8768730, 9.8618584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=63, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=17, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3344209, upper bound: 6.3344208
time: 13.89 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3344209, upper bound: 6.3344208
time: 28.18 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -26.9975948, -9.4375505, -26.9974194, -9.4884930, -11.9814949, 12.0338478
1: -9.8007765, 0.0243936, -9.8009510, -0.0089343, -6.8057861, 6.8482952
2: -4.3777714, 4.9003634, -4.3749690, 4.8851337, -6.8818245, 6.9122581
3: -13.4989452, -0.5680041, -13.4982166, -0.6284657, -9.6209335, 9.6824532
4: -5.5117984, 7.3962369, -5.5093565, 7.3381453, -8.9646339, 9.0215187
5: -8.9511557, 4.2512708, -8.9487495, 4.1937866, -11.5028152, 11.5633621
6: -24.1175499, -8.9015837, -24.1229210, -8.9168253, -9.8348770, 9.8613014
7: -9.7023296, 2.7203054, -9.7002726, 2.6898770, -8.9053612, 8.9353256
8: -12.3521271, 3.1260457, -12.3496952, 3.1011872, -9.4597321, 9.4805794
9: -7.1000962, 8.7000198, -7.1002989, 8.6221876, -10.8915253, 10.9703445
10: -7.0775261, 7.2695541, -7.0759916, 7.1769915, -11.5912819, 11.6889725
11: -4.7242823, 5.0715075, -4.6732073, 5.0687752, -8.2534828, 8.1977081
12: -16.8786983, -0.5770050, -16.8752728, -0.5815537, -11.4323158, 11.4569092
13: -21.3906345, -3.0483522, -21.3866005, -3.0883627, -14.4821396, 14.5318604
14: -22.9310856, -5.0287733, -22.9090366, -5.0329323, -16.8905640, 16.8846283
15: -9.0649633, 3.5292838, -9.0628920, 3.4747891, -9.8612442, 9.9147377
16: -9.6603956, 1.1955056, -9.6611271, 1.1412342, -10.1112251, 10.1675072
17: -20.8450165, -4.1497569, -20.8166466, -4.1522055, -13.7577972, 13.7383499
18: -3.2513053, 11.8091755, -3.2259772, 11.8067799, -11.1624565, 11.1391907
19: 1.7863088, 11.0959864, 1.8462522, 11.0970678, -9.2572327, 9.1910095
20: -0.8537982, 9.8920202, -0.8073525, 9.8926497, -10.7464476, 10.6993732
21: 0.6819832, 13.1284962, 0.7487276, 13.1280527, -12.3192520, 12.2476578
22: 1.9419641, 12.3000259, 1.9831247, 12.3000126, -8.4318047, 8.3855286
23: 0.1032729, 11.1065636, 0.2052239, 11.1076450, -9.6090508, 9.4997520
24: -5.4538269, 9.5274181, -5.3668313, 9.5278530, -12.1397858, 12.0487900
25: -4.4776626, 9.7638178, -4.3906317, 9.7622223, -11.9667664, 11.8783188
26: 2.9082074, 16.3132915, 2.9783378, 16.3124771, -13.4042702, 13.3349533
27: 0.0324826, 12.2296333, 0.0918977, 12.2305794, -10.4825897, 10.4245644
28: 0.6544650, 12.6661081, 0.7327020, 12.6648350, -11.5988922, 11.5183601
29: -0.5114094, 9.0158043, -0.4702656, 9.0125523, -6.7707005, 6.7307663
30: -4.0841002, 9.9894009, -4.0185556, 9.9877901, -12.7573318, 12.6921768
31: -3.1384983, 11.8268633, -3.0744185, 11.8252010, -11.4735336, 11.4091644
32: -19.0474091, -5.8251891, -19.0507164, -5.8386898, -9.4074287, 9.4408150
33: -38.4584198, -16.8316593, -38.4273109, -16.8344727, -15.9581909, 15.9144440
34: -37.8614960, -23.4108410, -37.8637466, -23.4183311, -10.3325882, 10.3475075
35: -29.0396633, -14.0506153, -29.0151176, -14.0529165, -11.8228149, 11.7963829
36: -22.0543804, -9.1946621, -22.0335941, -9.1949883, -9.2464981, 9.2323227
37: -39.6906586, -18.9760284, -39.6355209, -18.9780674, -15.7704163, 15.7188873
38: -36.0490952, -19.3279762, -36.0289650, -19.3358383, -14.5044937, 14.4969864
39: -38.4060135, -16.9139767, -38.3856125, -16.9144173, -14.8234329, 14.8146515
40: -34.3810310, -20.4297428, -34.3874779, -20.4556198, -8.6812592, 8.7239895
41: -21.2176743, -5.2808313, -21.2117920, -5.2907438, -12.3947144, 12.4058380
42: -23.4746590, -11.3885403, -23.4721584, -11.3910723, -9.8835220, 9.8656197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=63, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 659

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3463212, upper bound: 6.3471097
time: 11.78 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3463212, upper bound: 6.3463206
time: 12.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.75 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 26.75
Output dim: 26, lower bound: -6.3344209, upper bound: 6.3270271
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 26.75
Output dim: 26, lower bound: -6.3344209, upper bound: 6.3270271
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 26.75
Output dim: 26, lower bound: -6.3463212, upper bound: 6.3397723
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 26.75
Output dim: 26, lower bound: -6.3463212, upper bound: 6.3389907
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 26.75
Output dim: 26, lower bound: -6.3344209, upper bound: 6.3344208
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 26.75
Output dim: 26, lower bound: -6.3344209, upper bound: 6.3344208
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 26.75
Output dim: 26, lower bound: -6.3463212, upper bound: 6.3471097
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 26.75
Output dim: 26, lower bound: -6.3463212, upper bound: 6.3463206

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -26.9752769, -9.4555140, -26.9919739, -9.4902477, -11.9572067, 12.0119972
1: -9.7885857, 0.0108304, -9.7979126, -0.0100677, -6.7898827, 6.8262939
2: -4.3708224, 4.8953395, -4.3735714, 4.8845739, -6.8738022, 6.8978310
3: -13.4736519, -0.5868862, -13.4919558, -0.6290355, -9.5974808, 9.6569443
4: -5.4715238, 7.3774104, -5.4990644, 7.3376012, -8.9249573, 8.9910431
5: -8.9207478, 4.2345991, -8.9414139, 4.1933870, -11.4737396, 11.5362282
6: -24.0898876, -8.9233561, -24.1143970, -8.9176874, -9.8119278, 9.8308105
7: -9.6850176, 2.7093759, -9.6963043, 2.6895351, -8.8874359, 8.9188004
8: -12.3346624, 3.1202679, -12.3461456, 3.1005678, -9.4428501, 9.4695950
9: -7.0616994, 8.6832924, -7.0914040, 8.6213388, -10.8535538, 10.9438744
10: -7.0232496, 7.2410622, -7.0633268, 7.1756601, -11.5402489, 11.6450500
11: -4.7123799, 5.0527768, -4.6723022, 5.0642681, -8.2322235, 8.1770325
12: -16.8720284, -0.5919102, -16.8747692, -0.5845466, -11.4173813, 11.4317856
13: -21.3856049, -3.0688596, -21.3861809, -3.0927238, -14.4706039, 14.5091095
14: -22.9034233, -5.0591478, -22.9063377, -5.0413437, -16.8473663, 16.8587189
15: -9.0299463, 3.5114837, -9.0550146, 3.4732785, -9.8261375, 9.8873367
16: -9.6304932, 1.1787357, -9.6538944, 1.1402516, -10.0817070, 10.1448441
17: -20.8233871, -4.1904564, -20.8148174, -4.1623049, -13.7213974, 13.6963196
18: -3.2381632, 11.8015499, -3.2236392, 11.8060951, -11.1472015, 11.1267815
19: 1.8033690, 11.0718021, 1.8475566, 11.0907764, -9.2331657, 9.1659393
20: -0.8340180, 9.8805637, -0.8050408, 9.8893919, -10.7234097, 10.6856041
21: 0.7023635, 13.1107483, 0.7508752, 13.1230154, -12.2916412, 12.2181320
22: 1.9590859, 12.2826996, 1.9842138, 12.2954817, -8.4094238, 8.3669662
23: 0.1351757, 11.0534916, 0.2062283, 11.0929136, -9.5628052, 9.4509583
24: -5.4292612, 9.4892950, -5.3659849, 9.5175495, -12.1051025, 12.0139160
25: -4.4480996, 9.7132807, -4.3894949, 9.7489033, -11.9231682, 11.8287811
26: 2.9395652, 16.2955856, 2.9807730, 16.3072166, -13.3676510, 13.3148127
27: 0.0513632, 12.2094994, 0.0930824, 12.2250099, -10.4618378, 10.4104843
28: 0.6830432, 12.6233912, 0.7343030, 12.6532602, -11.5592194, 11.4762154
29: -0.4970293, 8.9956093, -0.4696149, 9.0074596, -6.7507706, 6.7109299
30: -4.0607080, 9.9582205, -4.0170674, 9.9793329, -12.7242279, 12.6634598
31: -3.1231289, 11.8049192, -3.0727952, 11.8202219, -11.4520760, 11.3818359
32: -19.0256042, -5.8433776, -19.0445442, -5.8398714, -9.3875389, 9.4173126
33: -38.4505005, -16.8548813, -38.4258957, -16.8388596, -15.9445267, 15.8894577
34: -37.8463440, -23.4299145, -37.8594894, -23.4207878, -10.3152657, 10.3254929
35: -29.0356312, -14.0747519, -29.0141163, -14.0573425, -11.8139229, 11.7733231
36: -22.0473328, -9.2208729, -22.0328674, -9.2005062, -9.2357903, 9.2059536
37: -39.6747169, -19.0046158, -39.6343613, -18.9843216, -15.7502975, 15.6893616
38: -36.0379524, -19.3559856, -36.0264015, -19.3404655, -14.4906845, 14.4596405
39: -38.3999062, -16.9353333, -38.3844261, -16.9184952, -14.8124924, 14.7853317
40: -34.3516350, -20.4492798, -34.3786621, -20.4561348, -8.6555672, 8.6943436
41: -21.2042274, -5.3002548, -21.2079277, -5.2934346, -12.3777084, 12.3790588
42: -23.4728699, -11.3951902, -23.4725418, -11.3927641, -9.8691597, 9.8573112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=62, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 595

## Relational analysis of IS_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3454172, upper bound: 6.3122875
time: 15.17 seconds

## Relational analysis of IS_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3455010, upper bound: 6.3389518
time: 11.03 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -26.9830208, -9.4294119, -26.9921093, -9.4900818, -11.9665337, 12.0454330
1: -9.7917318, 0.0279894, -9.7973967, -0.0099406, -6.7983189, 6.8494701
2: -4.3730650, 4.9028482, -4.3735867, 4.8846316, -6.8768845, 6.9068394
3: -13.4840879, -0.5655015, -13.4936171, -0.6290150, -9.6071091, 9.6803932
4: -5.4891834, 7.4040976, -5.5018320, 7.3374987, -8.9407425, 9.0189095
5: -8.9332857, 4.2546701, -8.9433413, 4.1934328, -11.4850922, 11.5582161
6: -24.0824471, -8.9233608, -24.1111031, -8.9174147, -9.8088760, 9.8300552
7: -9.6932430, 2.7287359, -9.6973257, 2.6895843, -8.8938675, 8.9400711
8: -12.3437786, 3.1329939, -12.3466873, 3.1004689, -9.4483910, 9.4740829
9: -7.0849667, 8.7205276, -7.0950780, 8.6213932, -10.8750610, 10.9846840
10: -7.0514917, 7.2854967, -7.0678225, 7.1754684, -11.5655022, 11.6939049
11: -4.7225437, 5.0622349, -4.6723118, 5.0654302, -8.2435951, 8.1865234
12: -16.8895321, -0.5840120, -16.8746662, -0.5834949, -11.4358788, 11.4374084
13: -21.4042358, -3.0594320, -21.3861046, -3.0924664, -14.4724808, 14.5227509
14: -22.9126854, -5.0507965, -22.9065285, -5.0405083, -16.8509445, 16.8692780
15: -9.0531273, 3.5365267, -9.0579262, 3.4732068, -9.8493385, 9.9154510
16: -9.6454840, 1.2200980, -9.6563120, 1.1404691, -10.0940208, 10.1892815
17: -20.8561802, -4.1693206, -20.8151283, -4.1590900, -13.7566910, 13.7151642
18: -3.2426472, 11.8096466, -3.2240469, 11.8061886, -11.1511307, 11.1355171
19: 1.7867486, 11.0825500, 1.8476315, 11.0924854, -9.2543106, 9.1760864
20: -0.8435290, 9.8818846, -0.8052397, 9.8889256, -10.7324543, 10.6871243
21: 0.6956315, 13.1112623, 0.7506821, 13.1215210, -12.3162079, 12.2288475
22: 1.9402604, 12.2883434, 1.9839878, 12.2962036, -8.4308357, 8.3735752
23: 0.1049535, 11.0726557, 0.2064345, 11.0960522, -9.5963211, 9.4662743
24: -5.4509330, 9.5040083, -5.3660545, 9.5196753, -12.1284332, 12.0278778
25: -4.4802580, 9.7326393, -4.3895726, 9.7519522, -11.9584274, 11.8463173
26: 2.9204068, 16.2949524, 2.9802790, 16.3058453, -13.3854389, 13.3146734
27: 0.0431786, 12.2165527, 0.0930369, 12.2257366, -10.4643402, 10.4158363
28: 0.6602876, 12.6395140, 0.7342846, 12.6556826, -11.5833359, 11.4909821
29: -0.5133561, 9.0025234, -0.4697344, 9.0084677, -6.7685051, 6.7175865
30: -4.0734544, 9.9678688, -4.0172172, 9.9804096, -12.7285690, 12.6665802
31: -3.1382036, 11.8176670, -3.0726089, 11.8214817, -11.4773483, 11.3938942
32: -19.0245094, -5.8406053, -19.0431480, -5.8395948, -9.3870468, 9.4189682
33: -38.4736290, -16.8390999, -38.4259644, -16.8365459, -15.9705353, 15.9047089
34: -37.8468094, -23.4217491, -37.8591232, -23.4199257, -10.3165207, 10.3351135
35: -29.0573406, -14.0577240, -29.0140305, -14.0547428, -11.8292999, 11.7869339
36: -22.0776119, -9.2022533, -22.0329075, -9.1975193, -9.2602539, 9.2215042
37: -39.7058907, -18.9882240, -39.6343765, -18.9817524, -15.7837677, 15.7043076
38: -36.0608597, -19.3395901, -36.0264969, -19.3379612, -14.5162506, 14.4752693
39: -38.4255066, -16.9204044, -38.3845978, -16.9161072, -14.8398285, 14.7987289
40: -34.3426247, -20.4499168, -34.3742599, -20.4560299, -8.6515884, 8.6952724
41: -21.2046165, -5.2926579, -21.2073460, -5.2925806, -12.3791199, 12.3844948
42: -23.4752827, -11.3908014, -23.4724407, -11.3924046, -9.8732834, 9.8616371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=62, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 595

## Relational analysis of IS_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3454172, upper bound: 6.3115060
time: 32.74 seconds

## Relational analysis of IS_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3455010, upper bound: 6.3381701
time: 11.63 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -26.9849033, -9.4397631, -26.9949188, -9.4889088, -11.9665680, 12.0292358
1: -9.7939215, 0.0227437, -9.7995834, -0.0092707, -6.7959328, 6.8450623
2: -4.3741980, 4.8995490, -4.3742695, 4.8849840, -6.8774414, 6.9105835
3: -13.4872417, -0.5686941, -13.4959593, -0.6286039, -9.6086731, 9.6792030
4: -5.4935665, 7.3956237, -5.5058184, 7.3380246, -8.9467697, 9.0168839
5: -8.9372492, 4.2508130, -8.9460850, 4.1936622, -11.4885178, 11.5599403
6: -24.1161194, -8.9044838, -24.1226616, -8.9173813, -9.8320198, 9.8580170
7: -9.6930742, 2.7196283, -9.6984835, 2.6897459, -8.8956451, 8.9326286
8: -12.3401508, 3.1255360, -12.3473368, 3.1010976, -9.4507408, 9.4780865
9: -7.0763149, 8.6992807, -7.0957050, 8.6220560, -10.8674698, 10.9650345
10: -7.0487118, 7.2682433, -7.0704546, 7.1767454, -11.5625572, 11.6822701
11: -4.7232246, 5.0634766, -4.6729927, 5.0672088, -8.2508812, 8.1895142
12: -16.8781548, -0.5861659, -16.8751602, -0.5833453, -11.4284973, 11.4452400
13: -21.3891907, -3.0596333, -21.3863411, -3.0906248, -14.4771690, 14.5260468
14: -22.9250603, -5.0386114, -22.9077930, -5.0349913, -16.8758087, 16.8778381
15: -9.0450449, 3.5280714, -9.0590382, 3.4745526, -9.8413124, 9.9093704
16: -9.6428127, 1.1936040, -9.6577358, 1.1408792, -10.0934105, 10.1628075
17: -20.8418331, -4.1712198, -20.8160210, -4.1563716, -13.7491150, 13.7164917
18: -3.2480521, 11.8076143, -3.2253382, 11.8065195, -11.1571121, 11.1359291
19: 1.7880151, 11.0849867, 1.8465908, 11.0949249, -9.2537003, 9.1786575
20: -0.8503091, 9.8877544, -0.8066700, 9.8918266, -10.7421360, 10.6944246
21: 0.6855824, 13.1233311, 0.7494297, 13.1268215, -12.3136063, 12.2331314
22: 1.9441276, 12.2930031, 1.9835577, 12.2986355, -8.4284248, 8.3772240
23: 0.1044314, 11.0869188, 0.2054431, 11.1038532, -9.6042099, 9.4798508
24: -5.4527292, 9.5128212, -5.3666086, 9.5250607, -12.1359253, 12.0340042
25: -4.4757433, 9.7440624, -4.3902731, 9.7584038, -11.9608307, 11.8580513
26: 2.9124990, 16.3090858, 2.9791842, 16.3115654, -13.3990669, 13.3299017
27: 0.0341890, 12.2221661, 0.0922124, 12.2291431, -10.4793930, 10.4190636
28: 0.6568131, 12.6502342, 0.7331369, 12.6617146, -11.5938263, 11.5021324
29: -0.5099065, 9.0088491, -0.4699478, 9.0112114, -6.7678051, 6.7233810
30: -4.0816374, 9.9790745, -4.0180674, 9.9857883, -12.7531204, 12.6858826
31: -3.1364975, 11.8164349, -3.0740159, 11.8231726, -11.4688988, 11.3948174
32: -19.0462456, -5.8282952, -19.0504932, -5.8392692, -9.4053764, 9.4378777
33: -38.4568138, -16.8478088, -38.4270096, -16.8376141, -15.9542007, 15.8981400
34: -37.8605728, -23.4181671, -37.8635330, -23.4197235, -10.3276901, 10.3421593
35: -29.0388489, -14.0685081, -29.0149574, -14.0563889, -11.8187943, 11.7809639
36: -22.0534573, -9.2149887, -22.0334148, -9.1989613, -9.2417068, 9.2147274
37: -39.6894989, -18.9939766, -39.6352692, -18.9815273, -15.7655945, 15.7002335
38: -36.0479012, -19.3459930, -36.0287361, -19.3393707, -14.4995499, 14.4786148
39: -38.4045792, -16.9311638, -38.3852730, -16.9177284, -14.8185959, 14.7974396
40: -34.3774261, -20.4311256, -34.3867989, -20.4559078, -8.6763077, 8.7214260
41: -21.2163277, -5.2887807, -21.2115326, -5.2922654, -12.3907547, 12.3969650
42: -23.4736633, -11.3926954, -23.4719830, -11.3918724, -9.8814888, 9.8604851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=62, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 595

## Relational analysis of IS_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3454172, upper bound: 6.3196233
time: 12.49 seconds

## Relational analysis of IS_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3455010, upper bound: 6.3462889
time: 11.94 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -26.9926167, -9.4136448, -26.9950695, -9.4887314, -11.9758949, 12.0626640
1: -9.7970800, 0.0399137, -9.7990828, -0.0091243, -6.8043842, 6.8682728
2: -4.3764057, 4.9070606, -4.3743048, 4.8850431, -6.8805313, 6.9195938
3: -13.4977093, -0.5472815, -13.4975891, -0.6285906, -9.6182938, 9.7026558
4: -5.5112028, 7.4222612, -5.5086055, 7.3379240, -8.9625359, 9.0447731
5: -8.9497671, 4.2708368, -8.9480009, 4.1937065, -11.4998703, 11.5819321
6: -24.1086864, -8.9044886, -24.1193314, -8.9170780, -9.8289452, 9.8572330
7: -9.7013054, 2.7390437, -9.6994934, 2.6898150, -8.9020576, 8.9539108
8: -12.3493509, 3.1382554, -12.3478489, 3.1010029, -9.4562569, 9.4826088
9: -7.0995660, 8.7365322, -7.0993519, 8.6220789, -10.8889656, 11.0058556
10: -7.0769343, 7.3127036, -7.0749578, 7.1765532, -11.5878563, 11.7310944
11: -4.7333350, 5.0729361, -4.6730022, 5.0683851, -8.2622643, 8.1989708
12: -16.8956146, -0.5783120, -16.8750954, -0.5822114, -11.4469910, 11.4508896
13: -21.4078159, -3.0501952, -21.3862667, -3.0903921, -14.4791031, 14.5396614
14: -22.9343491, -5.0302362, -22.9080086, -5.0341930, -16.8793411, 16.8883743
15: -9.0682545, 3.5531421, -9.0619659, 3.4744763, -9.8645172, 9.9374733
16: -9.6577950, 1.2350016, -9.6601620, 1.1410904, -10.1057281, 10.2072945
17: -20.8745747, -4.1501069, -20.8163128, -4.1531563, -13.7843704, 13.7353134
18: -3.2525005, 11.8157253, -3.2257223, 11.8066139, -11.1610451, 11.1446800
19: 1.7714138, 11.0957165, 1.8466570, 11.0966291, -9.2748375, 9.1888199
20: -0.8598237, 9.8890915, -0.8068717, 9.8913612, -10.7511845, 10.6959629
21: 0.6788287, 13.1238079, 0.7492633, 13.1253300, -12.3381805, 12.2438431
22: 1.9253016, 12.2986488, 1.9833241, 12.2993441, -8.4498482, 8.3838520
23: 0.0742035, 11.1061115, 0.2056142, 11.1069574, -9.6377335, 9.4951706
24: -5.4743776, 9.5275307, -5.3667159, 9.5271788, -12.1592407, 12.0479164
25: -4.5078926, 9.7634001, -4.3903685, 9.7614441, -11.9961014, 11.8755798
26: 2.8933058, 16.3084412, 2.9786658, 16.3101959, -13.4168901, 13.3297749
27: 0.0259504, 12.2292109, 0.0921717, 12.2298717, -10.4818573, 10.4243660
28: 0.6340947, 12.6663504, 0.7331681, 12.6641417, -11.6178970, 11.5169106
29: -0.5262493, 9.0157347, -0.4701120, 9.0122194, -6.7855721, 6.7300262
30: -4.0944090, 9.9887362, -4.0181818, 9.9869013, -12.7574768, 12.6890030
31: -3.1515596, 11.8291960, -3.0738213, 11.8244400, -11.4941864, 11.4068680
32: -19.0451279, -5.8255067, -19.0491047, -5.8389931, -9.4049034, 9.4395294
33: -38.4800148, -16.8320389, -38.4271088, -16.8352776, -15.9802170, 15.9133911
34: -37.8609962, -23.4099560, -37.8631630, -23.4188805, -10.3289528, 10.3517876
35: -29.0605297, -14.0514889, -29.0148849, -14.0537472, -11.8341904, 11.7945786
36: -22.0837402, -9.1963234, -22.0334549, -9.1959686, -9.2662010, 9.2303314
37: -39.7206078, -18.9775848, -39.6353149, -18.9789410, -15.7990570, 15.7151642
38: -36.0708008, -19.3295174, -36.0288277, -19.3368702, -14.5251007, 14.4943237
39: -38.4301682, -16.9162750, -38.3854752, -16.9153862, -14.8459473, 14.8108978
40: -34.3684502, -20.4317017, -34.3823853, -20.4557858, -8.6722946, 8.7223492
41: -21.2167168, -5.2811861, -21.2109146, -5.2914505, -12.3921356, 12.4024353
42: -23.4760990, -11.3882875, -23.4718876, -11.3915358, -9.8856163, 9.8648109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=62, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 595

## Relational analysis of IS_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3454172, upper bound: 6.3188346
time: 12.93 seconds

## Relational analysis of IS_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3455010, upper bound: 6.3454999
time: 11.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.42 seconds
IS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.42
Output dim: 26, lower bound: -6.3454172, upper bound: 6.3122875
IS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 26.42
Output dim: 26, lower bound: -6.3455010, upper bound: 6.3389518
IS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.42
Output dim: 26, lower bound: -6.3454172, upper bound: 6.3115060
IS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.42
Output dim: 26, lower bound: -6.3455010, upper bound: 6.3381701
IS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.42
Output dim: 26, lower bound: -6.3454172, upper bound: 6.3196233
IS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 26.42
Output dim: 26, lower bound: -6.3455010, upper bound: 6.3462889
IS_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.42
Output dim: 26, lower bound: -6.3454172, upper bound: 6.3188346
IS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.42
Output dim: 26, lower bound: -6.3455010, upper bound: 6.3454999

## BFS IS instance: IS_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -26.9662952, -9.4588928, -26.9912643, -9.4905281, -11.9457741, 12.0081863
1: -9.7824240, 0.0079479, -9.7974167, -0.0103049, -6.7826233, 6.8232956
2: -4.3684330, 4.8941031, -4.3733597, 4.8844633, -6.8673019, 6.8960094
3: -13.4610538, -0.5882587, -13.4909592, -0.6291809, -9.5844078, 9.6544762
4: -5.4571848, 7.3759861, -5.4979177, 7.3374777, -8.9106178, 8.9882393
5: -8.9078293, 4.2337909, -8.9403877, 4.1933064, -11.4604797, 11.5342979
6: -24.0886879, -8.9253120, -24.1143208, -8.9178410, -9.8099976, 9.8286591
7: -9.6801367, 2.7075791, -9.6959066, 2.6893926, -8.8825073, 8.9165535
8: -12.3304234, 3.1185424, -12.3457956, 3.1004188, -9.4376221, 9.4673595
9: -7.0476661, 8.6819887, -7.0902748, 8.6212320, -10.8390083, 10.9414940
10: -7.0079637, 7.2382398, -7.0621147, 7.1754303, -11.5245514, 11.6414299
11: -4.7102275, 5.0350461, -4.6721225, 5.0628462, -8.2290649, 8.1588516
12: -16.8626156, -0.5971980, -16.8739700, -0.5849861, -11.4061508, 11.4274521
13: -21.3541965, -3.0723228, -21.3837032, -3.0930128, -14.4392204, 14.5038643
14: -22.8928146, -5.0612612, -22.9054794, -5.0415316, -16.8378601, 16.8562775
15: -9.0140095, 3.5093067, -9.0537529, 3.4731042, -9.8091736, 9.8837166
16: -9.6265068, 1.1752367, -9.6535587, 1.1399660, -10.0725670, 10.1407127
17: -20.8123589, -4.1925893, -20.8139458, -4.1624393, -13.7096481, 13.6933594
18: -3.2356346, 11.7899532, -3.2234464, 11.8051929, -11.1441498, 11.1148491
19: 1.8056476, 11.0557652, 1.8477414, 11.0894871, -9.2301407, 9.1496620
20: -0.8295937, 9.8687162, -0.8046601, 9.8884201, -10.7180138, 10.6733761
21: 0.7069063, 13.0911016, 0.7512279, 13.1214590, -12.2865143, 12.1981773
22: 1.9624720, 12.2768507, 1.9844699, 12.2950354, -8.4055519, 8.3580399
23: 0.1371731, 11.0289774, 0.2064127, 11.0909452, -9.5590591, 9.4261055
24: -5.4266334, 9.4660578, -5.3657842, 9.5156708, -12.1008072, 11.9901276
25: -4.4453316, 9.6978149, -4.3892746, 9.7476635, -11.9190979, 11.8125954
26: 2.9458389, 16.2770844, 2.9812918, 16.3057327, -13.3598938, 13.2957926
27: 0.0539212, 12.1902046, 0.0932889, 12.2234592, -10.4578896, 10.3906975
28: 0.6863360, 12.6037827, 0.7345655, 12.6517067, -11.5548859, 11.4560966
29: -0.4948866, 8.9871006, -0.4694164, 9.0067759, -6.7477646, 6.7017536
30: -4.0577440, 9.9358902, -4.0168381, 9.9775333, -12.7199097, 12.6405640
31: -3.1203597, 11.7893448, -3.0725608, 11.8189793, -11.4481316, 11.3659668
32: -19.0188618, -5.8458562, -19.0438766, -5.8400850, -9.3786240, 9.4148369
33: -38.4471283, -16.8632317, -38.4256287, -16.8395576, -15.9412308, 15.8776779
34: -37.8438034, -23.4325943, -37.8592911, -23.4210014, -10.3133202, 10.3204880
35: -29.0317631, -14.0776701, -29.0137825, -14.0576210, -11.8107834, 11.7651672
36: -22.0387497, -9.2231369, -22.0321598, -9.2006903, -9.2318954, 9.2033768
37: -39.6718140, -19.0190468, -39.6341438, -18.9854393, -15.7466049, 15.6741714
38: -36.0260735, -19.3574295, -36.0254173, -19.3405418, -14.4874496, 14.4559402
39: -38.3915291, -16.9357567, -38.3837433, -16.9185028, -14.8054657, 14.7838211
40: -34.3505096, -20.4510269, -34.3785934, -20.4563026, -8.6541290, 8.6918697
41: -21.2031898, -5.3081551, -21.2078476, -5.2940631, -12.3755875, 12.3715210
42: -23.4719296, -11.4029303, -23.4724598, -11.3933964, -9.8676453, 9.8498878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=61, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3022042
time: 12.46 seconds

## Relational analysis of IS_A1_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3118309
time: 11.34 seconds

## BFS IS instance: IS_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -26.9751835, -9.4339809, -26.9913330, -9.4908609, -11.9524193, 12.0373344
1: -9.7886238, 0.0250621, -9.7974529, -0.0105195, -6.7884178, 6.8416653
2: -4.3718309, 4.8993759, -4.3732548, 4.8842964, -6.8671227, 6.9103241
3: -13.4735317, -0.5544972, -13.4912624, -0.6293025, -9.5965652, 9.6886215
4: -5.4729271, 7.4030576, -5.4980550, 7.3374205, -8.9263153, 9.0153046
5: -8.9221802, 4.2653651, -8.9406023, 4.1932993, -11.4738846, 11.5661163
6: -24.0919952, -8.9225178, -24.1134949, -8.9180012, -9.8144073, 9.8314323
7: -9.6870432, 2.7180438, -9.6958151, 2.6892166, -8.8896446, 8.9270744
8: -12.3378401, 3.1272798, -12.3457050, 3.1003022, -9.4437981, 9.4772816
9: -7.0613308, 8.7202616, -7.0906048, 8.6211319, -10.8530159, 10.9799995
10: -7.0231314, 7.2769389, -7.0625696, 7.1753435, -11.5394287, 11.6815147
11: -4.7535615, 5.0526261, -4.6720753, 5.0631967, -8.2723541, 8.1760635
12: -16.8716412, -0.5655489, -16.8736076, -0.5850281, -11.4133682, 11.4603195
13: -21.3836060, -2.9953289, -21.3844528, -3.0930963, -14.4678802, 14.5811539
14: -22.9099884, -5.0375757, -22.9053497, -5.0416994, -16.8551483, 16.8799667
15: -9.0315571, 3.5426667, -9.0536938, 3.4729688, -9.8271065, 9.9186287
16: -9.6312542, 1.1827688, -9.6535521, 1.1396174, -10.0726929, 10.1614456
17: -20.8258114, -4.1691651, -20.8136864, -4.1626005, -13.7241440, 13.7146301
18: -3.2682326, 11.8005886, -3.2232409, 11.8053932, -11.1770477, 11.1253014
19: 1.7636018, 11.0701876, 1.8478487, 11.0900002, -9.2724495, 9.1638260
20: -0.8635216, 9.8795185, -0.8044660, 9.8888264, -10.7523479, 10.6839848
21: 0.6521468, 13.1090508, 0.7514920, 13.1220131, -12.3417206, 12.2156372
22: 1.9434776, 12.2827806, 1.9848328, 12.2951317, -8.4304581, 8.3615532
23: 0.0763621, 11.0521860, 0.2064103, 11.0916719, -9.6203194, 9.4492264
24: -5.4875226, 9.4876394, -5.3656979, 9.5163918, -12.1621552, 12.0117607
25: -4.4816713, 9.7137566, -4.3891487, 9.7479868, -11.9556427, 11.8287430
26: 2.8933597, 16.2950058, 2.9815264, 16.3062725, -13.4129124, 13.3134794
27: 0.0020504, 12.2076626, 0.0935361, 12.2240715, -10.5105209, 10.4080276
28: 0.6372540, 12.6234999, 0.7346110, 12.6522388, -11.6042480, 11.4757309
29: -0.5187241, 8.9962339, -0.4692198, 9.0069828, -6.7718391, 6.7109814
30: -4.1138053, 9.9574251, -4.0167322, 9.9782152, -12.7760849, 12.6623459
31: -3.1660950, 11.8033590, -3.0723429, 11.8193235, -11.4938202, 11.3798218
32: -19.0276756, -5.8242922, -19.0439777, -5.8401079, -9.3867645, 9.4379234
33: -38.4691353, -16.8532887, -38.4255104, -16.8394299, -15.9760056, 15.8842621
34: -37.8447990, -23.4273987, -37.8582916, -23.4210758, -10.3223991, 10.3220024
35: -29.0423470, -14.0719452, -29.0137482, -14.0575886, -11.8382835, 11.7663155
36: -22.0453815, -9.2108326, -22.0307083, -9.2006989, -9.2404480, 9.2048264
37: -39.7087364, -19.0036106, -39.6340141, -18.9852524, -15.7849655, 15.6863174
38: -36.0352859, -19.3339329, -36.0230331, -19.3406048, -14.5104828, 14.4491882
39: -38.4058609, -16.9239883, -38.3839417, -16.9186020, -14.8206100, 14.7933502
40: -34.3535461, -20.4487743, -34.3783798, -20.4565544, -8.6567688, 8.6937008
41: -21.2184582, -5.2980852, -21.2077141, -5.2947283, -12.3860931, 12.3827820
42: -23.4800758, -11.3923054, -23.4722290, -11.3935823, -9.8756142, 9.8601036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=61, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A1_A2_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3288677
time: 11.02 seconds

## Relational analysis of IS_A1_A2_A1_A2_A2

### Relational analysis result of IS_A1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3384954
time: 13.77 seconds

## BFS IS instance: IS_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -26.9740562, -9.4328070, -26.9913673, -9.4903536, -11.9550781, 12.0416412
1: -9.7855930, 0.0251293, -9.7969084, -0.0101764, -6.7910652, 6.8464718
2: -4.3706627, 4.9016218, -4.3733969, 4.8845258, -6.8703842, 6.9050312
3: -13.4714823, -0.5668898, -13.4925909, -0.6291428, -9.5940018, 9.6779137
4: -5.4748354, 7.4026432, -5.5006819, 7.3373761, -8.9263802, 9.0161209
5: -8.9203815, 4.2537975, -8.9423237, 4.1933546, -11.4718628, 11.5563278
6: -24.0812759, -8.9253283, -24.1109943, -8.9175539, -9.8069344, 9.8278790
7: -9.6883593, 2.7269783, -9.6969357, 2.6894555, -8.8889389, 8.9378204
8: -12.3395777, 3.1312575, -12.3463545, 3.1003265, -9.4431553, 9.4718571
9: -7.0709200, 8.7192163, -7.0939240, 8.6212692, -10.8604965, 10.9823189
10: -7.0362029, 7.2827044, -7.0666122, 7.1752396, -11.5498657, 11.6902695
11: -4.7203646, 5.0444932, -4.6721458, 5.0640130, -8.2404442, 8.1683464
12: -16.8800888, -0.5893028, -16.8739090, -0.5838532, -11.4246559, 11.4330788
13: -21.3728390, -3.0629139, -21.3836708, -3.0927610, -14.4411011, 14.5174675
14: -22.9020882, -5.0528870, -22.9056683, -5.0406895, -16.8414078, 16.8668747
15: -9.0372248, 3.5343471, -9.0566730, 3.4730244, -9.8323593, 9.9118385
16: -9.6415100, 1.2166362, -9.6560001, 1.1401935, -10.0848618, 10.1851768
17: -20.8450928, -4.1714554, -20.8142185, -4.1592584, -13.7449493, 13.7122116
18: -3.2401514, 11.7980738, -3.2238433, 11.8052502, -11.1480865, 11.1235847
19: 1.7890151, 11.0665007, 1.8478131, 11.0912046, -9.2512894, 9.1597977
20: -0.8391390, 9.8700438, -0.8048930, 9.8879614, -10.7271004, 10.6749363
21: 0.7001832, 13.0915813, 0.7510402, 13.1199493, -12.3111420, 12.2088699
22: 1.9436622, 12.2824965, 1.9842496, 12.2957344, -8.4269714, 8.3646660
23: 0.1069386, 11.0481644, 0.2066078, 11.0940781, -9.5925713, 9.4414330
24: -5.4482894, 9.4807844, -5.3658619, 9.5178223, -12.1240845, 12.0040932
25: -4.4775057, 9.7171478, -4.3893528, 9.7506886, -11.9543762, 11.8301086
26: 2.9266438, 16.2764664, 2.9807572, 16.3043747, -13.3777313, 13.2957096
27: 0.0457482, 12.1972761, 0.0932245, 12.2241793, -10.4603615, 10.3960457
28: 0.6635685, 12.6199141, 0.7345586, 12.6541147, -11.5790329, 11.4708862
29: -0.5112040, 8.9940090, -0.4695687, 9.0077877, -6.7654991, 6.7084179
30: -4.0704870, 9.9455547, -4.0169554, 9.9786053, -12.7242279, 12.6436996
31: -3.1354175, 11.8021145, -3.0723732, 11.8202486, -11.4734306, 11.3779869
32: -19.0177689, -5.8431053, -19.0424881, -5.8397737, -9.3781471, 9.4164963
33: -38.4702377, -16.8474464, -38.4256859, -16.8371887, -15.9672623, 15.8929176
34: -37.8442421, -23.4244308, -37.8588600, -23.4201584, -10.3145905, 10.3301010
35: -29.0534630, -14.0606117, -29.0137596, -14.0549726, -11.8261757, 11.7787857
36: -22.0689945, -9.2044868, -22.0322208, -9.1977415, -9.2563515, 9.2189255
37: -39.7029648, -19.0026665, -39.6341324, -18.9828835, -15.7800674, 15.6891174
38: -36.0490036, -19.3409462, -36.0254974, -19.3381062, -14.5130157, 14.4715881
39: -38.4171143, -16.9208546, -38.3838844, -16.9161606, -14.8328247, 14.7972183
40: -34.3415375, -20.4516335, -34.3741989, -20.4561615, -8.6501617, 8.6927967
41: -21.2035789, -5.3005924, -21.2072678, -5.2932339, -12.3769989, 12.3769608
42: -23.4743748, -11.3985453, -23.4723797, -11.3930702, -9.8717499, 9.8542252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=61, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A1_A2_A2_A1_A1

### Relational analysis result of IS_A1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3014225
time: 11.18 seconds

## Relational analysis of IS_A1_A2_A2_A1_A2

### Relational analysis result of IS_A1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3110494
time: 11.85 seconds

## BFS IS instance: IS_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -26.9829254, -9.4078789, -26.9914379, -9.4906893, -11.9617424, 12.0707932
1: -9.7917786, 0.0422184, -9.7969484, -0.0103784, -6.7968616, 6.8648453
2: -4.3740273, 4.9068913, -4.3733072, 4.8843517, -6.8702087, 6.9193344
3: -13.4839916, -0.5330968, -13.4928846, -0.6292944, -9.6061668, 9.7120361
4: -5.4905715, 7.4296961, -5.5008106, 7.3373489, -8.9420929, 9.0431633
5: -8.9347000, 4.2854118, -8.9425306, 4.1933475, -11.4852371, 11.5881004
6: -24.0845680, -8.9225206, -24.1101990, -8.9177322, -9.8113518, 9.8306465
7: -9.6952648, 2.7374263, -9.6968603, 2.6893034, -8.8960991, 8.9483261
8: -12.3470030, 3.1400013, -12.3462629, 3.1002173, -9.4493332, 9.4817753
9: -7.0845938, 8.7575045, -7.0942531, 8.6211767, -10.8745193, 11.0208282
10: -7.0513697, 7.3214130, -7.0670600, 7.1751447, -11.5647583, 11.7303314
11: -4.7636881, 5.0620732, -4.6720982, 5.0643854, -8.2837219, 8.1855278
12: -16.8891258, -0.5577351, -16.8735161, -0.5838811, -11.4318390, 11.4659462
13: -21.4022617, -2.9858837, -21.3844528, -3.0929356, -14.4697800, 14.5947762
14: -22.9192505, -5.0292454, -22.9055634, -5.0408649, -16.8586807, 16.8905716
15: -9.0548019, 3.5677166, -9.0566006, 3.4728973, -9.8503380, 9.9467430
16: -9.6462479, 1.2241201, -9.6559849, 1.1398380, -10.0849609, 10.2058563
17: -20.8586044, -4.1480694, -20.8139668, -4.1594234, -13.7593918, 13.7334595
18: -3.2727742, 11.8087158, -3.2236466, 11.8054867, -11.1809807, 11.1340446
19: 1.7469838, 11.0809269, 1.8479083, 11.0917110, -9.2936096, 9.1739769
20: -0.8730392, 9.8808622, -0.8046749, 9.8883667, -10.7614059, 10.6855373
21: 0.6454167, 13.1095734, 0.7512906, 13.1205130, -12.3663330, 12.2263565
22: 1.9246893, 12.2884350, 1.9846048, 12.2958546, -8.4518471, 8.3681660
23: 0.0461171, 11.0713663, 0.2066067, 11.0947952, -9.6538200, 9.4645462
24: -5.5091953, 9.5023880, -5.3657589, 9.5185032, -12.1854706, 12.0256996
25: -4.5138364, 9.7330914, -4.3892326, 9.7509975, -11.9909019, 11.8462601
26: 2.8741670, 16.2943935, 2.9809866, 16.3048916, -13.4307251, 13.3134069
27: -0.0061462, 12.2147083, 0.0934637, 12.2248144, -10.5129776, 10.4133644
28: 0.6145229, 12.6396379, 0.7345822, 12.6546688, -11.6283493, 11.4905281
29: -0.5350189, 9.0031395, -0.4693754, 9.0079947, -6.7895470, 6.7176399
30: -4.1265688, 9.9670906, -4.0168304, 9.9793167, -12.7804260, 12.6655121
31: -3.1811514, 11.8161354, -3.0721529, 11.8205547, -11.5190964, 11.3918610
32: -19.0265865, -5.8215542, -19.0426064, -5.8398371, -9.3862801, 9.4395828
33: -38.4923058, -16.8375244, -38.4256287, -16.8370762, -16.0020294, 15.8995018
34: -37.8452797, -23.4192066, -37.8578987, -23.4202137, -10.3236618, 10.3316345
35: -29.0640411, -14.0549383, -29.0136890, -14.0549736, -11.8536758, 11.7799187
36: -22.0756607, -9.1922092, -22.0307617, -9.1977415, -9.2648964, 9.2203579
37: -39.7398453, -18.9872093, -39.6339989, -18.9826946, -15.8184280, 15.7012634
38: -36.0581741, -19.3174591, -36.0231171, -19.3381004, -14.5360947, 14.4648209
39: -38.4314499, -16.9091148, -38.3840714, -16.9162140, -14.8479843, 14.8067627
40: -34.3445969, -20.4493885, -34.3739281, -20.4564552, -8.6527977, 8.6946373
41: -21.2188473, -5.2904782, -21.2071495, -5.2938223, -12.3874969, 12.3882332
42: -23.4825039, -11.3878841, -23.4721489, -11.3932018, -9.8797302, 9.8644562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=61, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A1_A2_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3280858
time: 14.21 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3377138
time: 13.37 seconds

## BFS IS instance: IS_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -26.9758930, -9.4431419, -26.9941864, -9.4891968, -11.9551239, 12.0254555
1: -9.7877703, 0.0198870, -9.7990866, -0.0094967, -6.7886734, 6.8420734
2: -4.3717895, 4.8983088, -4.3740692, 4.8848820, -6.8709373, 6.9087620
3: -13.4746466, -0.5700867, -13.4949179, -0.6287096, -9.5955925, 9.6767311
4: -5.4792476, 7.3941817, -5.5046735, 7.3379259, -8.9324417, 9.0141144
5: -8.9243698, 4.2499743, -8.9450369, 4.1936207, -11.4752960, 11.5580368
6: -24.1149464, -8.9064283, -24.1225719, -8.9175453, -9.8300896, 9.8558636
7: -9.6882038, 2.7178612, -9.6980944, 2.6896105, -8.8907051, 8.9303856
8: -12.3359413, 3.1237833, -12.3469810, 3.1009264, -9.4455338, 9.4758530
9: -7.0622711, 8.6979580, -7.0945959, 8.6219425, -10.8529167, 10.9626656
10: -7.0334387, 7.2654324, -7.0692201, 7.1765318, -11.5468941, 11.6786308
11: -4.7210560, 5.0457616, -4.6728377, 5.0657978, -8.2477379, 8.1713333
12: -16.8686962, -0.5914582, -16.8743706, -0.5837367, -11.4172478, 11.4409256
13: -21.3578224, -3.0630999, -21.3838215, -3.0909204, -14.4458008, 14.5207405
14: -22.9144878, -5.0407143, -22.9069519, -5.0351667, -16.8663101, 16.8753815
15: -9.0291500, 3.5259094, -9.0577669, 3.4743881, -9.8243523, 9.9057655
16: -9.6388340, 1.1901512, -9.6574154, 1.1405902, -10.0843010, 10.1587067
17: -20.8308067, -4.1733351, -20.8150845, -4.1565423, -13.7373428, 13.7135544
18: -3.2455461, 11.7960377, -3.2251332, 11.8055725, -11.1540794, 11.1240005
19: 1.7903063, 11.0689354, 1.8467705, 11.0936413, -9.2506714, 9.1623611
20: -0.8459082, 9.8759375, -0.8063097, 9.8908625, -10.7367706, 10.6822472
21: 0.6901085, 13.1036663, 0.7498176, 13.1252508, -12.3084946, 12.2131729
22: 1.9475031, 12.2871656, 1.9838285, 12.2981510, -8.4245796, 8.3683167
23: 0.1064181, 11.0624323, 0.2055851, 11.1018782, -9.6004982, 9.4550018
24: -5.4500527, 9.4895992, -5.3664217, 9.5232153, -12.1316147, 12.0102005
25: -4.4729781, 9.7285652, -4.3900537, 9.7571955, -11.9567566, 11.8418694
26: 2.9187679, 16.2905884, 2.9796796, 16.3100891, -13.3913212, 13.3109093
27: 0.0367615, 12.2028580, 0.0924199, 12.2276077, -10.4753876, 10.3992691
28: 0.6601005, 12.6306372, 0.7334328, 12.6601467, -11.5895081, 11.4820328
29: -0.5077776, 9.0003529, -0.4697868, 9.0105391, -6.7647972, 6.7142010
30: -4.0787039, 9.9567366, -4.0178275, 9.9839754, -12.7487640, 12.6630020
31: -3.1336992, 11.8008785, -3.0737858, 11.8219414, -11.4649734, 11.3789444
32: -19.0395203, -5.8307600, -19.0498180, -5.8394823, -9.3964691, 9.4354019
33: -38.4534874, -16.8561440, -38.4267464, -16.8382893, -15.9509277, 15.8863449
34: -37.8579979, -23.4208298, -37.8633423, -23.4199677, -10.3257408, 10.3371429
35: -29.0349674, -14.0714207, -29.0146484, -14.0566034, -11.8156433, 11.7728310
36: -22.0448704, -9.2172337, -22.0327148, -9.1991234, -9.2378082, 9.2121563
37: -39.6865616, -19.0084000, -39.6350365, -18.9826756, -15.7619171, 15.6850433
38: -36.0360870, -19.3474026, -36.0277519, -19.3394623, -14.4963074, 14.4749489
39: -38.3961563, -16.9316063, -38.3846664, -16.9177666, -14.8115845, 14.7959213
40: -34.3763657, -20.4328327, -34.3867340, -20.4560318, -8.6748695, 8.7189598
41: -21.2152843, -5.2966814, -21.2114067, -5.2929182, -12.3886490, 12.3894577
42: -23.4727135, -11.4004345, -23.4719086, -11.3924856, -9.8799934, 9.8530502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=61, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_A2_A1_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3095420
time: 13.31 seconds

## Relational analysis of IS_A2_A2_A1_A1_A2

### Relational analysis result of IS_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3191650
time: 13.15 seconds

## BFS IS instance: IS_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -26.9847908, -9.4181995, -26.9942760, -9.4895411, -11.9617805, 12.0545845
1: -9.7939701, 0.0369871, -9.7991447, -0.0097065, -6.7944736, 6.8604355
2: -4.3751850, 4.9035540, -4.3739719, 4.8847136, -6.8707542, 6.9230709
3: -13.4871292, -0.5363128, -13.4952374, -0.6288753, -9.6077499, 9.7108803
4: -5.4949527, 7.4212561, -5.5048151, 7.3378901, -8.9481506, 9.0411491
5: -8.9386892, 4.2815580, -8.9452229, 4.1936159, -11.4886475, 11.5897903
6: -24.1182251, -8.9036751, -24.1217537, -8.9176893, -9.8345070, 9.8586159
7: -9.6950979, 2.7282872, -9.6979933, 2.6894407, -8.8978500, 8.9408875
8: -12.3433752, 3.1325371, -12.3469009, 3.1008019, -9.4516945, 9.4857807
9: -7.0759444, 8.7362518, -7.0948877, 8.6218166, -10.8669052, 11.0011559
10: -7.0485945, 7.3041487, -7.0696487, 7.1764183, -11.5617943, 11.7187157
11: -4.7643423, 5.0633302, -4.6727657, 5.0661402, -8.2910500, 8.1885338
12: -16.8777580, -0.5597796, -16.8740139, -0.5837523, -11.4244766, 11.4738007
13: -21.3871727, -2.9860725, -21.3846130, -3.0910826, -14.4744835, 14.5980453
14: -22.9316406, -5.0170345, -22.9068508, -5.0353642, -16.8835449, 16.8990860
15: -9.0466919, 3.5592523, -9.0576906, 3.4742532, -9.8423004, 9.9406509
16: -9.6435604, 1.1976519, -9.6574097, 1.1402142, -10.0843925, 10.1794052
17: -20.8442440, -4.1499023, -20.8148518, -4.1566877, -13.7518311, 13.7347946
18: -3.2781544, 11.8066654, -3.2249205, 11.8057804, -11.1869659, 11.1344528
19: 1.7482791, 11.0833731, 1.8468618, 11.0941429, -9.2929878, 9.1765404
20: -0.8798029, 9.8867369, -0.8061225, 9.8912601, -10.7710629, 10.6928596
21: 0.6353588, 13.1216431, 0.7500844, 13.1258059, -12.3637085, 12.2306442
22: 1.9285212, 12.2930832, 1.9841785, 12.2982779, -8.4494400, 8.3718281
23: 0.0455935, 11.0856380, 0.2056062, 11.1025887, -9.6617508, 9.4781265
24: -5.5109501, 9.5111532, -5.3663392, 9.5238962, -12.1929779, 12.0318260
25: -4.5092936, 9.7445068, -4.3899031, 9.7574778, -11.9933243, 11.8580322
26: 2.8663020, 16.3085175, 2.9799094, 16.3106213, -13.4443188, 13.3286076
27: -0.0151551, 12.2203121, 0.0926478, 12.2282429, -10.5280457, 10.4166107
28: 0.6110454, 12.6503420, 0.7334960, 12.6606770, -11.6388397, 11.5016594
29: -0.5315918, 9.0094757, -0.4695708, 9.0107002, -6.7888603, 6.7234230
30: -4.1347528, 9.9782658, -4.0177336, 9.9846697, -12.8049545, 12.6848221
31: -3.1794598, 11.8149109, -3.0735750, 11.8222733, -11.5106544, 11.3927994
32: -19.0483208, -5.8092194, -19.0499306, -5.8395233, -9.4046021, 9.4584961
33: -38.4755096, -16.8462830, -38.4266396, -16.8381596, -15.9857025, 15.8929443
34: -37.8590164, -23.4156532, -37.8623428, -23.4200058, -10.3348045, 10.3386688
35: -29.0455551, -14.0657406, -29.0145950, -14.0566158, -11.8431625, 11.7739601
36: -22.0514946, -9.2049408, -22.0312614, -9.1991234, -9.2463608, 9.2135811
37: -39.7234573, -18.9929829, -39.6349602, -18.9824982, -15.8002701, 15.6971741
38: -36.0452499, -19.3239365, -36.0253868, -19.3394890, -14.5193787, 14.4681892
39: -38.4105225, -16.9198914, -38.3847885, -16.9178505, -14.8266983, 14.8054504
40: -34.3793793, -20.4306011, -34.3865204, -20.4563198, -8.6775093, 8.7208061
41: -21.2305489, -5.2866011, -21.2113113, -5.2935271, -12.3991623, 12.4007225
42: -23.4808826, -11.3897705, -23.4716682, -11.3926573, -9.8879433, 9.8632774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=61, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_A2_A1_A2_A1

### Relational analysis result of IS_A2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3362064
time: 9.97 seconds

## Relational analysis of IS_A2_A2_A1_A2_A2

### Relational analysis result of IS_A2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3458311
time: 13.22 seconds

## BFS IS instance: IS_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -26.9836693, -9.4170523, -26.9943199, -9.4890070, -11.9644585, 12.0588875
1: -9.7909241, 0.0370698, -9.7985725, -0.0093524, -6.7971191, 6.8652782
2: -4.3740387, 4.9058161, -4.3741345, 4.8849654, -6.8740273, 6.9177799
3: -13.4850912, -0.5486591, -13.4965706, -0.6286609, -9.6052094, 9.7001839
4: -5.4968925, 7.4208527, -5.5074501, 7.3378286, -8.9482155, 9.0419884
5: -8.9368858, 4.2700024, -8.9469662, 4.1936398, -11.4866562, 11.5800171
6: -24.1075134, -8.9064455, -24.1192703, -8.9172688, -9.8270111, 9.8550777
7: -9.6964340, 2.7372561, -9.6991253, 2.6896782, -8.8971481, 8.9516602
8: -12.3451366, 3.1364989, -12.3475218, 3.1008542, -9.4510384, 9.4803829
9: -7.0855269, 8.7351961, -7.0982242, 8.6219893, -10.8744125, 11.0034790
10: -7.0616441, 7.3099022, -7.0737162, 7.1763101, -11.5721893, 11.7274857
11: -4.7311802, 5.0552120, -4.6728339, 5.0669551, -8.2591286, 8.1807785
12: -16.8861980, -0.5835989, -16.8743095, -0.5826817, -11.4357529, 11.4465637
13: -21.3764076, -3.0536551, -21.3837852, -3.0906811, -14.4476967, 14.5343742
14: -22.9237289, -5.0323534, -22.9071598, -5.0343323, -16.8698578, 16.8859406
15: -9.0523396, 3.5509734, -9.0606737, 3.4742813, -9.8475342, 9.9338722
16: -9.6538391, 1.2315402, -9.6598282, 1.1408100, -10.0965729, 10.2031441
17: -20.8635139, -4.1522326, -20.8153763, -4.1533389, -13.7726212, 13.7323761
18: -3.2500155, 11.8041725, -3.2255101, 11.8056870, -11.1579971, 11.1327553
19: 1.7736897, 11.0796776, 1.8468461, 11.0953445, -9.2718086, 9.1725197
20: -0.8554327, 9.8772440, -0.8065157, 9.8904057, -10.7458382, 10.6837597
21: 0.6833713, 13.1041603, 0.7496145, 13.1237488, -12.3330994, 12.2238464
22: 1.9286785, 12.2927971, 1.9836001, 12.2988815, -8.4459953, 8.3749256
23: 0.0761669, 11.0815945, 0.2057881, 11.1050196, -9.6340256, 9.4703369
24: -5.4717331, 9.5042982, -5.3664756, 9.5253162, -12.1549072, 12.0241547
25: -4.5051441, 9.7479258, -4.3901443, 9.7602110, -11.9920197, 11.8593826
26: 2.8995600, 16.2899513, 2.9791746, 16.3087120, -13.4091520, 13.3107767
27: 0.0285676, 12.2099171, 0.0923676, 12.2283354, -10.4778748, 10.4045792
28: 0.6373646, 12.6467657, 0.7334282, 12.6625566, -11.6135864, 11.4968071
29: -0.5240936, 9.0072136, -0.4699161, 9.0115299, -6.7825546, 6.7208557
30: -4.0914688, 9.9664450, -4.0179682, 9.9850864, -12.7531509, 12.6660995
31: -3.1487780, 11.8136301, -3.0735793, 11.8231754, -11.4902725, 11.3909950
32: -19.0384121, -5.8279920, -19.0484467, -5.8391809, -9.3959961, 9.4370613
33: -38.4766617, -16.8403473, -38.4268417, -16.8359566, -15.9769135, 15.9016342
34: -37.8584480, -23.4126282, -37.8629761, -23.4191093, -10.3269806, 10.3467827
35: -29.0566692, -14.0544119, -29.0145969, -14.0540009, -11.8310585, 11.7864532
36: -22.0751553, -9.1986094, -22.0327492, -9.1961575, -9.2622757, 9.2277336
37: -39.7177582, -18.9919930, -39.6350632, -18.9801064, -15.7953720, 15.6999893
38: -36.0589523, -19.3309002, -36.0278587, -19.3369637, -14.5218506, 14.4906578
39: -38.4217834, -16.9167137, -38.3847847, -16.9154358, -14.8389282, 14.8093643
40: -34.3673363, -20.4335060, -34.3823013, -20.4559364, -8.6708794, 8.7198906
41: -21.2156715, -5.2890816, -21.2108364, -5.2921023, -12.3900452, 12.3949127
42: -23.4751453, -11.3960171, -23.4718227, -11.3921595, -9.8840942, 9.8573875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=61, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3087528
time: 12.01 seconds

## Relational analysis of IS_A2_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3183765
time: 23.43 seconds

## BFS IS instance: IS_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -26.9925365, -9.3920851, -26.9944057, -9.4893446, -11.9711037, 12.0880432
1: -9.7971344, 0.0541205, -9.7986374, -0.0095477, -6.8029289, 6.8836327
2: -4.3774195, 4.9110732, -4.3740177, 4.8847904, -6.8738403, 6.9320831
3: -13.4975891, -0.5149231, -13.4968891, -0.6288414, -9.6173401, 9.7343140
4: -5.5126238, 7.4479122, -5.5075679, 7.3377919, -8.9638939, 9.0690155
5: -8.9512339, 4.3015914, -8.9471817, 4.1936202, -11.5000458, 11.6118164
6: -24.1107998, -8.9036646, -24.1184673, -8.9174051, -9.8314171, 9.8578434
7: -9.7033310, 2.7476740, -9.6990194, 2.6895208, -8.9042892, 8.9621506
8: -12.3525696, 3.1452563, -12.3474054, 3.1007006, -9.4571953, 9.4902973
9: -7.0992112, 8.7734928, -7.0985327, 8.6218643, -10.8884315, 11.0419884
10: -7.0768409, 7.3486018, -7.0741506, 7.1762295, -11.5870705, 11.7675323
11: -4.7745123, 5.0727768, -4.6727695, 5.0673213, -8.3024063, 8.1980095
12: -16.8952026, -0.5519114, -16.8739510, -0.5826789, -11.4429283, 11.4794197
13: -21.4058533, -2.9766302, -21.3845844, -3.0908074, -14.4764023, 14.6116562
14: -22.9409294, -5.0086851, -22.9070587, -5.0345268, -16.8871155, 16.9096298
15: -9.0698833, 3.5843139, -9.0606165, 3.4741697, -9.8654976, 9.9687729
16: -9.6585464, 1.2389965, -9.6598358, 1.1404581, -10.0966911, 10.2238388
17: -20.8769779, -4.1288352, -20.8151474, -4.1534715, -13.7870941, 13.7536087
18: -3.2826278, 11.8147831, -3.2252889, 11.8058910, -11.1908798, 11.1432037
19: 1.7316551, 11.0941200, 1.8469515, 11.0958462, -9.3141098, 9.1867027
20: -0.8893123, 9.8880672, -0.8063331, 9.8908186, -10.7801304, 10.6944008
21: 0.6286280, 13.1221199, 0.7498839, 13.1243057, -12.3882751, 12.2413368
22: 1.9097290, 12.2987289, 1.9839249, 12.2989998, -8.4708519, 8.3784428
23: 0.0153942, 11.1048117, 0.2057993, 11.1057081, -9.6952438, 9.4934425
24: -5.5326605, 9.5259066, -5.3663840, 9.5260372, -12.2162781, 12.0457573
25: -4.5414438, 9.7638702, -4.3900042, 9.7605028, -12.0285645, 11.8755379
26: 2.8471146, 16.3078690, 2.9793921, 16.3092518, -13.4621372, 13.3284769
27: -0.0233114, 12.2273769, 0.0925829, 12.2289743, -10.5305023, 10.4219208
28: 0.5883248, 12.6664953, 0.7334504, 12.6631107, -11.6629181, 11.5164413
29: -0.5479286, 9.0163517, -0.4697059, 9.0117149, -6.8066025, 6.7300854
30: -4.1475029, 9.9879541, -4.0178432, 9.9857836, -12.8093262, 12.6879120
31: -3.1945074, 11.8276501, -3.0733688, 11.8235121, -11.5359306, 11.4048729
32: -19.0472260, -5.8064699, -19.0485477, -5.8392363, -9.4041290, 9.4601440
33: -38.4986458, -16.8305206, -38.4267578, -16.8358421, -16.0116959, 15.9082031
34: -37.8594513, -23.4074440, -37.8619347, -23.4191551, -10.3360748, 10.3482895
35: -29.0672112, -14.0487003, -29.0145397, -14.0539846, -11.8585701, 11.7875633
36: -22.0818043, -9.1863060, -22.0313091, -9.1961384, -9.2708321, 9.2291698
37: -39.7546158, -18.9765854, -39.6349335, -18.9799614, -15.8337021, 15.7120895
38: -36.0681267, -19.3074188, -36.0254440, -19.3370171, -14.5449142, 14.4838715
39: -38.4361229, -16.9049740, -38.3849297, -16.9154587, -14.8540497, 14.8188705
40: -34.3703880, -20.4312363, -34.3820724, -20.4561958, -8.6735001, 8.7217274
41: -21.2309399, -5.2790060, -21.2107239, -5.2926998, -12.4005356, 12.4061852
42: -23.4832840, -11.3853779, -23.4715996, -11.3923073, -9.8920593, 9.8676300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=61, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3354171
time: 11.35 seconds

## Relational analysis of IS_A2_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3450425
time: 12.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.36 seconds
IS_A1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3022042
IS_A1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3118309
IS_A1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3288677
IS_A1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3384954
IS_A1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3014225
IS_A1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3110494
IS_A1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3280858
IS_A1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3377138
IS_A2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3095420
IS_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3191650
IS_A2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3362064
IS_A2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3458311
IS_A2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3087528
IS_A2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3449598, upper bound: 6.3183765
IS_A2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3354171
IS_A2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.36
Output dim: 26, lower bound: -6.3450437, upper bound: 6.3450425

## BFS IS instance: IS_A1_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -26.9540367, -9.4608307, -26.9877129, -9.4911032, -11.9322052, 12.0030823
1: -9.7718363, 0.0065320, -9.7943563, -0.0107098, -6.7715588, 6.8191223
2: -4.3600597, 4.8932538, -4.3709688, 4.8842168, -6.8585739, 6.8927879
3: -13.4473686, -0.5892379, -13.4870319, -0.6294594, -9.5708122, 9.6497078
4: -5.4390059, 7.3752632, -5.4927106, 7.3372717, -8.8920937, 8.9820709
5: -8.8951359, 4.2329788, -8.9367409, 4.1930819, -11.4477463, 11.5299377
6: -24.0864239, -8.9298763, -24.1136513, -8.9191418, -9.8005867, 9.8222504
7: -9.6669064, 2.7070827, -9.6921196, 2.6892376, -8.8694916, 8.9123306
8: -12.3154221, 3.1177077, -12.3415270, 3.1002142, -9.4230042, 9.4625587
9: -7.0260715, 8.6809902, -7.0840816, 8.6209469, -10.8170395, 10.9342117
10: -6.9967995, 7.2364845, -7.0589399, 7.1749344, -11.5129776, 11.6367378
11: -4.7088614, 5.0233073, -4.6717234, 5.0594854, -8.2245369, 8.1466522
12: -16.8614426, -0.6053835, -16.8736763, -0.5873106, -11.4023895, 11.4224319
13: -21.3389397, -3.0746756, -21.3792648, -3.0936632, -14.4228821, 14.4972725
14: -22.8895950, -5.0654507, -22.9045792, -5.0427551, -16.8329849, 16.8525238
15: -9.0058880, 3.5071139, -9.0514164, 3.4724681, -9.7998695, 9.8787384
16: -9.6120167, 1.1740723, -9.6493950, 1.1396372, -10.0577164, 10.1357574
17: -20.8096638, -4.2084279, -20.8131561, -4.1669860, -13.7023544, 13.6775818
18: -3.2332795, 11.7807999, -3.2227905, 11.8025532, -11.1391907, 11.1050835
19: 1.8070428, 11.0477390, 1.8481443, 11.0871925, -9.2264175, 9.1401863
20: -0.8266196, 9.8650227, -0.8038073, 9.8873625, -10.7139816, 10.6688299
21: 0.7096179, 13.0856514, 0.7520092, 13.1198854, -12.2815323, 12.1878242
22: 1.9638834, 12.2663164, 1.9849043, 12.2920027, -8.4014168, 8.3473701
23: 0.1384861, 11.0135746, 0.2067955, 11.0865526, -9.5534286, 9.4101334
24: -5.4250827, 9.4464436, -5.3653498, 9.5100603, -12.0937653, 11.9702110
25: -4.4436765, 9.6791477, -4.3888140, 9.7423391, -11.9121170, 11.7935410
26: 2.9489017, 16.2673225, 2.9821606, 16.3029251, -13.3540230, 13.2851620
27: 0.0554500, 12.1810188, 0.0937190, 12.2208309, -10.4538879, 10.3816223
28: 0.6884847, 12.5900135, 0.7351639, 12.6477470, -11.5490646, 11.4413109
29: -0.4942019, 8.9736271, -0.4692168, 9.0029306, -6.7432766, 6.6882954
30: -4.0557003, 9.9177856, -4.0162692, 9.9723663, -12.7127991, 12.6219940
31: -3.1189423, 11.7763205, -3.0721781, 11.8152637, -11.4429359, 11.3523903
32: -19.0154266, -5.8472738, -19.0429001, -5.8404779, -9.3699150, 9.4108810
33: -38.4447403, -16.8705902, -38.4249344, -16.8416672, -15.9369354, 15.8680038
34: -37.8422089, -23.4431915, -37.8588181, -23.4240303, -10.3094406, 10.3111343
35: -29.0300732, -14.0843172, -29.0133286, -14.0594988, -11.8076019, 11.7578506
36: -22.0373001, -9.2263908, -22.0317535, -9.2016296, -9.2257004, 9.1973362
37: -39.6700172, -19.0300407, -39.6336327, -18.9886131, -15.7414322, 15.6621094
38: -36.0231857, -19.3593941, -36.0245895, -19.3411217, -14.4843674, 14.4531517
39: -38.3871002, -16.9361591, -38.3824425, -16.9186935, -14.7999649, 14.7813416
40: -34.3483276, -20.4518185, -34.3779373, -20.4565239, -8.6479225, 8.6876793
41: -21.2006035, -5.3124247, -21.2071095, -5.2952833, -12.3653183, 12.3646545
42: -23.4704037, -11.4097424, -23.4720306, -11.3953943, -9.8633156, 9.8405800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A1_A1_A1_B1

### Relational analysis result of IS_A1_A2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3010856
time: 14.33 seconds

## Relational analysis of IS_A1_A2_A1_A1_A1_B2

### Relational analysis result of IS_A1_A2_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3010856
time: 13.10 seconds

## BFS IS instance: IS_A1_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -26.9664516, -9.4466505, -26.9909439, -9.4906492, -11.9940529, 12.0202065
1: -9.7819853, 0.0189679, -9.7971354, -0.0103726, -6.8156490, 6.8340282
2: -4.3685436, 4.9044490, -4.3731680, 4.8844037, -6.8776512, 6.9063034
3: -13.4607201, -0.5710230, -13.4906559, -0.6292419, -9.5840340, 9.6716309
4: -5.4571519, 7.3950109, -5.4974899, 7.3374453, -8.9244041, 9.0067635
5: -8.9078255, 4.2498059, -8.9400778, 4.1932955, -11.4614563, 11.5503922
6: -24.0958786, -8.9243488, -24.1141758, -8.9179525, -9.8022003, 9.8339043
7: -9.6798820, 2.7208757, -9.6955795, 2.6893644, -8.8873940, 8.9295692
8: -12.3301334, 3.1323059, -12.3454609, 3.1003797, -9.4444828, 9.4802799
9: -7.0469985, 8.7042103, -7.0897999, 8.6211653, -10.8549347, 10.9624748
10: -7.0101357, 7.2498384, -7.0618811, 7.1753330, -11.5248184, 11.6432838
11: -4.7222939, 5.0357456, -4.6720333, 5.0625949, -8.2374191, 8.1579933
12: -16.8757210, -0.5969926, -16.8739071, -0.5852014, -11.4091530, 11.4304085
13: -21.3539505, -3.0521679, -21.3833447, -3.0931416, -14.4456673, 14.5231590
14: -22.9040298, -5.0607758, -22.9052715, -5.0416737, -16.8657532, 16.8581161
15: -9.0140533, 3.5198243, -9.0535498, 3.4729788, -9.8077126, 9.8885384
16: -9.6260796, 1.1858687, -9.6532230, 1.1399074, -10.0877037, 10.1512184
17: -20.8283806, -4.1927433, -20.8138008, -4.1628213, -13.7187119, 13.6908798
18: -3.2521524, 11.7901525, -3.2233157, 11.8049717, -11.1605148, 11.1088829
19: 1.7996154, 11.0554295, 1.8478260, 11.0893250, -9.2363129, 9.1624413
20: -0.8373811, 9.8687963, -0.8045437, 9.8883457, -10.7257271, 10.6733398
21: 0.7012179, 13.0913153, 0.7514603, 13.1213417, -12.2988892, 12.1982918
22: 1.9516444, 12.2763624, 1.9845886, 12.2947807, -8.4163589, 8.3718281
23: 0.1196477, 11.0286255, 0.2064497, 11.0906000, -9.5762711, 9.4331741
24: -5.4459333, 9.4655447, -5.3656812, 9.5152416, -12.1196899, 11.9886513
25: -4.4644384, 9.6977272, -4.3891459, 9.7472305, -11.9378014, 11.8177872
26: 2.9304295, 16.2772942, 2.9813924, 16.3054943, -13.3750648, 13.2959023
27: 0.0407624, 12.1900730, 0.0934596, 12.2232685, -10.4676590, 10.3868942
28: 0.6731265, 12.6036263, 0.7346492, 12.6513519, -11.5678253, 11.4711723
29: -0.5077001, 8.9882193, -0.4693315, 9.0064754, -6.7602978, 6.7120247
30: -4.0765672, 9.9353094, -4.0166674, 9.9771433, -12.7383575, 12.6480331
31: -3.1349804, 11.7889862, -3.0724459, 11.8187037, -11.4621735, 11.3778381
32: -19.0209465, -5.8460426, -19.0432816, -5.8401470, -9.3774986, 9.4202118
33: -38.4559937, -16.8615665, -38.4255180, -16.8396988, -15.9482193, 15.8781090
34: -37.8568230, -23.4330025, -37.8592033, -23.4212608, -10.3191795, 10.3016701
35: -29.0373631, -14.0755434, -29.0137329, -14.0577564, -11.8163605, 11.7661972
36: -22.0393982, -9.2231188, -22.0318604, -9.2008038, -9.2495880, 9.2034760
37: -39.6856003, -19.0181732, -39.6340103, -18.9856949, -15.7499008, 15.6713638
38: -36.0268211, -19.3589668, -36.0243988, -19.3406525, -14.4737091, 14.4536552
39: -38.3901062, -16.9364586, -38.3822250, -16.9185257, -14.7701416, 14.7803879
40: -34.3522377, -20.4523354, -34.3784866, -20.4568329, -8.6553001, 8.6906281
41: -21.2064285, -5.3066816, -21.2077255, -5.2942915, -12.3715820, 12.3793335
42: -23.4789963, -11.4018297, -23.4724350, -11.3936529, -9.8590317, 9.8477249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=163, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A1_A1_A2_B1

### Relational analysis result of IS_A1_A2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3107124
time: 13.20 seconds

## Relational analysis of IS_A1_A2_A1_A1_A2_B2

### Relational analysis result of IS_A1_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3107124
time: 13.51 seconds

## BFS IS instance: IS_A1_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -26.9629288, -9.4359007, -26.9878159, -9.4914312, -11.9388542, 12.0322189
1: -9.7780190, 0.0236487, -9.7944145, -0.0109150, -6.7773628, 6.8374901
2: -4.3634524, 4.8985205, -4.3708553, 4.8840437, -6.8583870, 6.9071045
3: -13.4598503, -0.5554790, -13.4873409, -0.6295838, -9.5829353, 9.6838264
4: -5.4547300, 7.4023476, -5.4928393, 7.3372235, -8.9078026, 9.0091248
5: -8.9094715, 4.2645855, -8.9369621, 4.1930890, -11.4611359, 11.5617027
6: -24.0897064, -8.9270992, -24.1128235, -8.9193325, -9.8050041, 9.8250198
7: -9.6738138, 2.7175465, -9.6920357, 2.6890755, -8.8766136, 8.9228249
8: -12.3228722, 3.1264591, -12.3414364, 3.1000359, -9.4291744, 9.4724751
9: -7.0397620, 8.7192535, -7.0844154, 8.6208401, -10.8310547, 10.9727058
10: -7.0119839, 7.2751961, -7.0593472, 7.1748466, -11.5278549, 11.6767616
11: -4.7521801, 5.0408611, -4.6716948, 5.0598507, -8.2678528, 8.1638565
12: -16.8704929, -0.5737147, -16.8732700, -0.5873735, -11.4095612, 11.4553375
13: -21.3683624, -2.9976349, -21.3800545, -3.0937843, -14.4515457, 14.5746002
14: -22.9067726, -5.0417976, -22.9044533, -5.0429173, -16.8502121, 16.8761902
15: -9.0234022, 3.5404882, -9.0513506, 3.4723499, -9.8178101, 9.9136543
16: -9.6167488, 1.1815848, -9.6493950, 1.1392703, -10.0578194, 10.1564713
17: -20.8230934, -4.1850672, -20.8128967, -4.1671538, -13.7168198, 13.6987762
18: -3.2658546, 11.7914429, -3.2225597, 11.8027744, -11.1720543, 11.1155167
19: 1.7650108, 11.0621643, 1.8482552, 11.0876970, -9.2687416, 9.1543465
20: -0.8605554, 9.8758259, -0.8036067, 9.8877687, -10.7483244, 10.6794329
21: 0.6548243, 13.1036091, 0.7522969, 13.1204548, -12.3367157, 12.2053070
22: 1.9448948, 12.2722712, 1.9852467, 12.2920990, -8.4263229, 8.3508625
23: 0.0776751, 11.0367918, 0.2068005, 11.0872536, -9.6146851, 9.4332733
24: -5.4860120, 9.4680414, -5.3652563, 9.5107546, -12.1551132, 11.9918327
25: -4.4800329, 9.6950836, -4.3886614, 9.7426300, -11.9486885, 11.8096581
26: 2.8964057, 16.2852669, 2.9824195, 16.3034554, -13.4070492, 13.3028469
27: 0.0035579, 12.1984959, 0.0939503, 12.2214813, -10.5065422, 10.3989563
28: 0.6394153, 12.6097422, 0.7352033, 12.6483078, -11.5983734, 11.4609337
29: -0.5180527, 8.9827576, -0.4690201, 9.0031090, -6.7673531, 6.6975269
30: -4.1117902, 9.9393015, -4.0161419, 9.9730511, -12.7690048, 12.6438293
31: -3.1646976, 11.7903471, -3.0719495, 11.8155870, -11.4886246, 11.3662376
32: -19.0242729, -5.8257098, -19.0430031, -5.8405099, -9.3780518, 9.4339790
33: -38.4668159, -16.8607597, -38.4248161, -16.8415585, -15.9717484, 15.8745613
34: -37.8432236, -23.4379768, -37.8578186, -23.4241123, -10.3185196, 10.3126450
35: -29.0406437, -14.0786152, -29.0132580, -14.0595150, -11.8351288, 11.7589722
36: -22.0439224, -9.2140942, -22.0302925, -9.2016439, -9.2342377, 9.1987762
37: -39.7069130, -19.0145950, -39.6334610, -18.9884586, -15.7797852, 15.6742325
38: -36.0323944, -19.3359299, -36.0221939, -19.3411484, -14.5074234, 14.4463730
39: -38.4014664, -16.9244423, -38.3825874, -16.9187012, -14.8150558, 14.7908630
40: -34.3513603, -20.4495163, -34.3777657, -20.4567928, -8.6505585, 8.6895142
41: -21.2158775, -5.3023663, -21.2069626, -5.2959180, -12.3758163, 12.3759232
42: -23.4785652, -11.3990793, -23.4718037, -11.3955421, -9.8713074, 9.8507881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3277490
time: 11.95 seconds

## Relational analysis of IS_A1_A2_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3277490
time: 13.57 seconds

## BFS IS instance: IS_A1_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -26.9753494, -9.4216728, -26.9910069, -9.4909973, -12.0006866, 12.0493469
1: -9.7881861, 0.0360770, -9.7971935, -0.0105786, -6.8214340, 6.8523998
2: -4.3719239, 4.9096804, -4.3730717, 4.8842378, -6.8774567, 6.9206200
3: -13.4731607, -0.5372391, -13.4909668, -0.6293800, -9.5961914, 9.7057762
4: -5.4728847, 7.4220791, -5.4976163, 7.3373680, -8.9401016, 9.0338135
5: -8.9221725, 4.2814279, -8.9402790, 4.1932797, -11.4748535, 11.5821953
6: -24.0991554, -8.9215803, -24.1133347, -8.9181156, -9.8066101, 9.8366833
7: -9.6867609, 2.7313614, -9.6954889, 2.6892118, -8.8945503, 8.9400902
8: -12.3375378, 3.1410823, -12.3453379, 3.1002471, -9.4506645, 9.4902229
9: -7.0606818, 8.7425156, -7.0901151, 8.6210451, -10.8689308, 11.0009537
10: -7.0253363, 7.2885389, -7.0623198, 7.1752481, -11.5396957, 11.6833611
11: -4.7656250, 5.0533090, -4.6719971, 5.0629516, -8.2807083, 8.1751633
12: -16.8847885, -0.5653509, -16.8735123, -0.5852586, -11.4163818, 11.4632568
13: -21.3833103, -2.9750986, -21.3841095, -3.0932703, -14.4743271, 14.6004677
14: -22.9212151, -5.0371008, -22.9051380, -5.0418730, -16.8829880, 16.8817902
15: -9.0315628, 3.5531774, -9.0534887, 3.4728475, -9.8256416, 9.9234657
16: -9.6308231, 1.1933632, -9.6532183, 1.1395564, -10.0878296, 10.1719551
17: -20.8418922, -4.1693420, -20.8135529, -4.1629548, -13.7331924, 13.7121582
18: -3.2847641, 11.8008175, -3.2230911, 11.8051872, -11.1934204, 11.1193123
19: 1.7575946, 11.0698748, 1.8479326, 11.0898304, -9.2786217, 9.1766052
20: -0.8712788, 9.8795948, -0.8043385, 9.8887529, -10.7600317, 10.6839333
21: 0.6464400, 13.1092978, 0.7517192, 13.1218967, -12.3541031, 12.2157669
22: 1.9326496, 12.2823133, 1.9849415, 12.2948999, -8.4412651, 8.3753414
23: 0.0588508, 11.0518513, 0.2064741, 11.0913258, -9.6375198, 9.4562798
24: -5.5068550, 9.4871416, -5.3656092, 9.5159540, -12.1810684, 12.0102577
25: -4.5007524, 9.7136869, -4.3890152, 9.7475624, -11.9743805, 11.8339500
26: 2.8779402, 16.2951851, 2.9816194, 16.3060341, -13.4280939, 13.3135662
27: -0.0111406, 12.2075186, 0.0936882, 12.2238894, -10.5203094, 10.4042358
28: 0.6240723, 12.6233273, 0.7346835, 12.6519136, -11.6171494, 11.4908066
29: -0.5315304, 8.9973364, -0.4691331, 9.0066662, -6.7843685, 6.7212467
30: -4.1326628, 9.9568281, -4.0165582, 9.9778242, -12.7945328, 12.6698456
31: -3.1807277, 11.8029947, -3.0722291, 11.8190403, -11.5078659, 11.3916969
32: -19.0297546, -5.8244872, -19.0433655, -5.8401690, -9.3856468, 9.4433289
33: -38.4780045, -16.8516045, -38.4253845, -16.8396034, -15.9829941, 15.8846893
34: -37.8578110, -23.4278297, -37.8582001, -23.4213257, -10.3282433, 10.3032188
35: -29.0479546, -14.0698557, -29.0136604, -14.0577517, -11.8438454, 11.7673111
36: -22.0459976, -9.2108269, -22.0303745, -9.2007942, -9.2581177, 9.2049236
37: -39.7224998, -19.0027523, -39.6339111, -18.9855499, -15.7882767, 15.6834717
38: -36.0359879, -19.3354759, -36.0220108, -19.3406773, -14.4967499, 14.4468651
39: -38.4044418, -16.9246902, -38.3824043, -16.9186382, -14.7852478, 14.7899017
40: -34.3552780, -20.4500694, -34.3783112, -20.4570751, -8.6579285, 8.6924648
41: -21.2217312, -5.2966356, -21.2076187, -5.2948885, -12.3820801, 12.3905869
42: -23.4871693, -11.3911905, -23.4721985, -11.3938236, -9.8670044, 9.8579292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=163, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3373769
time: 14.36 seconds

## Relational analysis of IS_A1_A2_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3373769
time: 11.24 seconds

## BFS IS instance: IS_A1_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -26.9618111, -9.4347296, -26.9878330, -9.4909096, -11.9415321, 12.0365410
1: -9.7749691, 0.0237021, -9.7938547, -0.0105844, -6.7799988, 6.8423157
2: -4.3622732, 4.9007683, -4.3709898, 4.8843026, -6.8616409, 6.9018116
3: -13.4578238, -0.5678620, -13.4886456, -0.6294224, -9.5803986, 9.6731300
4: -5.4566522, 7.4019346, -5.4954901, 7.3371687, -8.9078712, 9.0099335
5: -8.9076681, 4.2530041, -8.9386673, 4.1931362, -11.4591293, 11.5519180
6: -24.0789967, -8.9298830, -24.1103382, -8.9188614, -9.7975502, 9.8214893
7: -9.6751451, 2.7264953, -9.6931477, 2.6893263, -8.8759155, 8.9335861
8: -12.3245955, 3.1304312, -12.3420420, 3.1001015, -9.4285316, 9.4670448
9: -7.0493546, 8.7182026, -7.0877280, 8.6209955, -10.8385315, 10.9750328
10: -7.0250506, 7.2809534, -7.0634012, 7.1747293, -11.5382576, 11.6855545
11: -4.7190256, 5.0327454, -4.6717458, 5.0606394, -8.2359200, 8.1561203
12: -16.8789158, -0.5975574, -16.8735943, -0.5862513, -11.4208755, 11.4280968
13: -21.3575687, -3.0653353, -21.3792458, -3.0934286, -14.4247665, 14.5108833
14: -22.8988934, -5.0571432, -22.9047775, -5.0419378, -16.8365021, 16.8631058
15: -9.0290833, 3.5321722, -9.0543108, 3.4724021, -9.8230820, 9.9068604
16: -9.6269941, 1.2154727, -9.6518126, 1.1398444, -10.0699921, 10.1801987
17: -20.8424149, -4.1873169, -20.8134537, -4.1638007, -13.7376480, 13.6964111
18: -3.2377412, 11.7888994, -3.2231266, 11.8026314, -11.1431122, 11.1138153
19: 1.7903962, 11.0584803, 1.8482239, 11.0888987, -9.2475586, 9.1503448
20: -0.8361697, 9.8663435, -0.8040144, 9.8869038, -10.7230740, 10.6703577
21: 0.7028944, 13.0861416, 0.7518208, 13.1183977, -12.3061447, 12.1985016
22: 1.9450736, 12.2719316, 1.9846630, 12.2927170, -8.4228363, 8.3539886
23: 0.1082323, 11.0327511, 0.2069725, 11.0896683, -9.5869446, 9.4254799
24: -5.4468079, 9.4611626, -5.3654089, 9.5122089, -12.1170654, 11.9841461
25: -4.4758563, 9.6985035, -4.3888903, 9.7453699, -11.9473953, 11.8110466
26: 2.9296885, 16.2666874, 2.9816561, 16.3015442, -13.3718557, 13.2850313
27: 0.0472560, 12.1881046, 0.0936751, 12.2215557, -10.4563293, 10.3869362
28: 0.6657171, 12.6061478, 0.7351725, 12.6501760, -11.5731964, 11.4560852
29: -0.5105487, 8.9805431, -0.4693692, 9.0039310, -6.7610207, 6.6949444
30: -4.0684772, 9.9274492, -4.0163798, 9.9734497, -12.7171631, 12.6251144
31: -3.1340125, 11.7890787, -3.0719795, 11.8165159, -11.4682388, 11.3644142
32: -19.0143452, -5.8445230, -19.0415077, -5.8402100, -9.3694191, 9.4125137
33: -38.4678879, -16.8548565, -38.4250145, -16.8392868, -15.9629440, 15.8832397
34: -37.8426971, -23.4349709, -37.8584442, -23.4231720, -10.3107109, 10.3207512
35: -29.0517693, -14.0673161, -29.0132675, -14.0568886, -11.8230209, 11.7714539
36: -22.0675564, -9.2077694, -22.0317879, -9.1986685, -9.2501488, 9.2128582
37: -39.7011719, -19.0136375, -39.6336098, -18.9860420, -15.7748642, 15.6770477
38: -36.0460968, -19.3429375, -36.0246887, -19.3386669, -14.5099182, 14.4687958
39: -38.4126930, -16.9212761, -38.3826065, -16.9162903, -14.8273163, 14.7947159
40: -34.3393936, -20.4524536, -34.3735313, -20.4563732, -8.6439476, 8.6885929
41: -21.2010117, -5.3047962, -21.2065086, -5.2944698, -12.3667297, 12.3701019
42: -23.4728203, -11.4053221, -23.4719543, -11.3950434, -9.8674316, 9.8449059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A2_A1_A1_B1

### Relational analysis result of IS_A1_A2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3004058
time: 11.10 seconds

## Relational analysis of IS_A1_A2_A2_A1_A1_B2

### Relational analysis result of IS_A1_A2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3004058
time: 10.62 seconds

## BFS IS instance: IS_A1_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -26.9741669, -9.4205446, -26.9910507, -9.4904861, -12.0033379, 12.0536652
1: -9.7851486, 0.0361199, -9.7966223, -0.0102310, -6.8240757, 6.8572140
2: -4.3707719, 4.9119682, -4.3732166, 4.8844824, -6.8807449, 6.9153309
3: -13.4711475, -0.5496492, -13.4922943, -0.6291995, -9.5936623, 9.6950684
4: -5.4748116, 7.4216623, -5.5002604, 7.3373599, -8.9401817, 9.0346336
5: -8.9203587, 4.2698307, -8.9420366, 4.1933298, -11.4728470, 11.5724297
6: -24.0884666, -8.9243727, -24.1108646, -8.9176788, -9.7991257, 9.8331451
7: -9.6881046, 2.7402668, -9.6966076, 2.6894298, -8.8938255, 8.9508324
8: -12.3392868, 3.1450140, -12.3459673, 3.1003122, -9.4500160, 9.4847832
9: -7.0702648, 8.7414722, -7.0934639, 8.6212006, -10.8764076, 11.0032959
10: -7.0384121, 7.2943234, -7.0663719, 7.1751213, -11.5501060, 11.6921120
11: -4.7324333, 5.0451941, -4.6720495, 5.0637674, -8.2487831, 8.1674805
12: -16.8931980, -0.5891533, -16.8738804, -0.5840951, -11.4276962, 11.4360085
13: -21.3725471, -3.0427475, -21.3833160, -3.0929022, -14.4475021, 14.5367813
14: -22.9133530, -5.0524445, -22.9054642, -5.0408783, -16.8692398, 16.8686981
15: -9.0372448, 3.5448663, -9.0564432, 3.4728861, -9.8309212, 9.9166527
16: -9.6410685, 1.2272596, -9.6556492, 1.1401227, -10.1000023, 10.1956673
17: -20.8611946, -4.1716304, -20.8141098, -4.1596222, -13.7539825, 13.7097397
18: -3.2566385, 11.7982616, -3.2237046, 11.8050442, -11.1644630, 11.1175995
19: 1.7829654, 11.0661888, 1.8479054, 11.0910282, -9.2574539, 9.1725845
20: -0.8469038, 9.8701239, -0.8047516, 9.8878880, -10.7347918, 10.6748753
21: 0.6944740, 13.0918121, 0.7512774, 13.1198330, -12.3235168, 12.2089806
22: 1.9328432, 12.2819853, 1.9843702, 12.2955017, -8.4377708, 8.3784447
23: 0.0894160, 11.0478191, 0.2066616, 11.0937328, -9.6097679, 9.4484787
24: -5.4676137, 9.4802656, -5.3657799, 9.5173931, -12.1430130, 12.0025787
25: -4.4965820, 9.7170858, -4.3892407, 9.7502956, -11.9731140, 11.8353119
26: 2.9112430, 16.2766609, 2.9809155, 16.3041306, -13.3928871, 13.2957458
27: 0.0325825, 12.1971474, 0.0934029, 12.2239876, -10.4701157, 10.3922615
28: 0.6503823, 12.6197643, 0.7346621, 12.6538038, -11.5919418, 11.4859467
29: -0.5240293, 8.9951134, -0.4694670, 9.0074978, -6.7780342, 6.7186756
30: -4.0893393, 9.9449863, -4.0168114, 9.9782209, -12.7426758, 12.6511765
31: -3.1500430, 11.8017387, -3.0722415, 11.8199530, -11.4874725, 11.3898544
32: -19.0198593, -5.8432646, -19.0418854, -5.8398581, -9.3770370, 9.4218636
33: -38.4791718, -16.8457737, -38.4255981, -16.8373909, -15.9742584, 15.8933258
34: -37.8572693, -23.4248390, -37.8587799, -23.4204273, -10.3204422, 10.3112755
35: -29.0590782, -14.0585251, -29.0136852, -14.0551329, -11.8317413, 11.7798042
36: -22.0696526, -9.2045059, -22.0318985, -9.1978388, -9.2739563, 9.2190266
37: -39.7168121, -19.0017853, -39.6340370, -18.9831238, -15.7833633, 15.6863098
38: -36.0497017, -19.3424988, -36.0244560, -19.3381729, -14.4992905, 14.4692841
39: -38.4156990, -16.9215393, -38.3823814, -16.9161968, -14.7975388, 14.7938156
40: -34.3432884, -20.4529495, -34.3740692, -20.4567032, -8.6513176, 8.6915627
41: -21.2068405, -5.2990680, -21.2071419, -5.2934408, -12.3729782, 12.3847733
42: -23.4814529, -11.3974276, -23.4723320, -11.3933144, -9.8631363, 9.8520660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=163, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A2_A1_A2_B1

### Relational analysis result of IS_A1_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3100329
time: 16.32 seconds

## Relational analysis of IS_A1_A2_A2_A1_A2_B2

### Relational analysis result of IS_A1_A2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3100329
time: 13.44 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -26.9706593, -9.4098129, -26.9879189, -9.4912338, -11.9481659, 12.0656891
1: -9.7812004, 0.0407906, -9.7939167, -0.0107875, -6.7857990, 6.8606720
2: -4.3656559, 4.9060354, -4.3708930, 4.8841295, -6.8614731, 6.9161148
3: -13.4703159, -0.5340726, -13.4889708, -0.6295366, -9.5925522, 9.7072716
4: -5.4723935, 7.4289947, -5.4955893, 7.3371058, -8.9235954, 9.0369987
5: -8.9220018, 4.2846026, -8.9388676, 4.1931229, -11.4725037, 11.5837135
6: -24.0822868, -8.9270916, -24.1095390, -8.9190397, -9.8019600, 9.8242397
7: -9.6820335, 2.7369213, -9.6930571, 2.6891575, -8.8830566, 8.9440842
8: -12.3320274, 3.1391518, -12.3419561, 3.0999818, -9.4347095, 9.4769650
9: -7.0630159, 8.7565031, -7.0880423, 8.6208706, -10.8525505, 11.0135384
10: -7.0402246, 7.3196383, -7.0638628, 7.1746397, -11.5531425, 11.7256088
11: -4.7623420, 5.0503197, -4.6717095, 5.0610180, -8.2792358, 8.1733131
12: -16.8879490, -0.5658525, -16.8732300, -0.5862619, -11.4280510, 11.4609413
13: -21.3869667, -2.9882050, -21.3800507, -3.0935373, -14.4534531, 14.5882187
14: -22.9160156, -5.0334377, -22.9046440, -5.0421009, -16.8537521, 16.8868027
15: -9.0466290, 3.5655487, -9.0542574, 3.4722643, -9.8410492, 9.9417877
16: -9.6317215, 1.2229490, -9.6518183, 1.1394985, -10.0701065, 10.2009048
17: -20.8558426, -4.1639299, -20.8131943, -4.1639833, -13.7521210, 13.7176285
18: -3.2703011, 11.7995796, -3.2229362, 11.8028679, -11.1759949, 11.1242714
19: 1.7483678, 11.0729170, 1.8483243, 11.0894117, -9.2898827, 9.1645164
20: -0.8700852, 9.8771305, -0.8038323, 9.8873091, -10.7573948, 10.6809626
21: 0.6480877, 13.1041260, 0.7520857, 13.1189375, -12.3613358, 12.2159996
22: 1.9261107, 12.2779102, 1.9850254, 12.2928295, -8.4477196, 8.3574963
23: 0.0474453, 11.0559788, 0.2069885, 11.0903625, -9.6482010, 9.4485970
24: -5.5077009, 9.4827852, -5.3653250, 9.5129128, -12.1784363, 12.0057564
25: -4.5121965, 9.7144651, -4.3887472, 9.7456722, -11.9839478, 11.8272209
26: 2.8772244, 16.2846317, 2.9818835, 16.3020897, -13.4248657, 13.3027477
27: -0.0046129, 12.2055779, 0.0939152, 12.2221928, -10.5089722, 10.4042892
28: 0.6166618, 12.6258631, 0.7352128, 12.6507378, -11.6224899, 11.4757004
29: -0.5343657, 8.9896507, -0.4691690, 9.0041218, -6.7850685, 6.7041798
30: -4.1245365, 9.9489851, -4.0162449, 9.9741068, -12.7733536, 12.6469193
31: -3.1797602, 11.8030920, -3.0717294, 11.8168344, -11.5138931, 11.3782616
32: -19.0231819, -5.8229713, -19.0416088, -5.8402181, -9.3775673, 9.4356422
33: -38.4899292, -16.8449249, -38.4249039, -16.8391876, -15.9977341, 15.8898239
34: -37.8436890, -23.4298019, -37.8574257, -23.4232521, -10.3197670, 10.3222847
35: -29.0623512, -14.0616150, -29.0131874, -14.0568800, -11.8505211, 11.7725868
36: -22.0742016, -9.1954851, -22.0303268, -9.1986732, -9.2586937, 9.2143097
37: -39.7381287, -18.9982147, -39.6334953, -18.9858551, -15.8132324, 15.6891479
38: -36.0553207, -19.3194618, -36.0222855, -19.3386822, -14.5330048, 14.4619904
39: -38.4270630, -16.9095459, -38.3827171, -16.9163742, -14.8424225, 14.8042526
40: -34.3424072, -20.4501801, -34.3732986, -20.4566402, -8.6465759, 8.6904297
41: -21.2162838, -5.2947264, -21.2063904, -5.2950754, -12.3772125, 12.3813934
42: -23.4809761, -11.3946686, -23.4717026, -11.3951855, -9.8754082, 9.8551407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A2_A2_A1_B1

### Relational analysis result of IS_A1_A2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3270691
time: 14.86 seconds

## Relational analysis of IS_A1_A2_A2_A2_A1_B2

### Relational analysis result of IS_A1_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443854, upper bound: 6.3270691
time: 13.55 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -26.9830780, -9.3956032, -26.9911194, -9.4908161, -12.0099983, 12.0828171
1: -9.7913456, 0.0532188, -9.7966919, -0.0104384, -6.8298779, 6.8755741
2: -4.3741550, 4.9172182, -4.3731041, 4.8843050, -6.8805428, 6.9296303
3: -13.4836292, -0.5158253, -13.4926109, -0.6293461, -9.6058083, 9.7292137
4: -5.4905481, 7.4487219, -5.5003586, 7.3372951, -8.9558945, 9.0616951
5: -8.9347258, 4.3014288, -8.9422054, 4.1933160, -11.4862289, 11.6042023
6: -24.0917473, -8.9215717, -24.1100540, -8.9178324, -9.8035431, 9.8358974
7: -9.6950092, 2.7507443, -9.6965179, 2.6892648, -8.9009743, 8.9613495
8: -12.3466845, 3.1537902, -12.3458681, 3.1001675, -9.4561844, 9.4947300
9: -7.0839581, 8.7797709, -7.0937405, 8.6211014, -10.8904228, 11.0417786
10: -7.0535583, 7.3330107, -7.0667953, 7.1750274, -11.5649681, 11.7321892
11: -4.7757635, 5.0627651, -4.6720266, 5.0641222, -8.2920799, 8.1846504
12: -16.9022331, -0.5574887, -16.8734531, -0.5841361, -11.4348831, 11.4688797
13: -21.4019508, -2.9657083, -21.3841057, -3.0930276, -14.4762115, 14.6140900
14: -22.9304619, -5.0287189, -22.9053516, -5.0410509, -16.8865051, 16.8923798
15: -9.0548010, 3.5782447, -9.0564003, 3.4727745, -9.8488579, 9.9515800
16: -9.6458168, 1.2347360, -9.6556339, 1.1397552, -10.1001129, 10.2163658
17: -20.8746452, -4.1482329, -20.8138237, -4.1598234, -13.7684631, 13.7309570
18: -3.2892518, 11.8089333, -3.2235003, 11.8052807, -11.1973534, 11.1280670
19: 1.7409568, 11.0806351, 1.8480060, 11.0915375, -9.2997742, 9.1867676
20: -0.8808305, 9.8809156, -0.8045428, 9.8882866, -10.7691174, 10.6854582
21: 0.6397033, 13.1098003, 0.7515383, 13.1203794, -12.3787079, 12.2264709
22: 1.9138527, 12.2879667, 1.9847212, 12.2956209, -8.4626503, 8.3819332
23: 0.0286207, 11.0710239, 0.2066721, 11.0944424, -9.6710281, 9.4716072
24: -5.5285420, 9.5018549, -5.3656864, 9.5180883, -12.2043610, 12.0241661
25: -4.5329342, 9.7330513, -4.3891168, 9.7505627, -12.0096588, 11.8514671
26: 2.8587546, 16.2945805, 2.9811201, 16.3046684, -13.4459133, 13.3134604
27: -0.0193181, 12.2145920, 0.0936344, 12.2246227, -10.5227814, 10.4095726
28: 0.6013246, 12.6394653, 0.7346747, 12.6543350, -11.6412659, 11.5055847
29: -0.5478220, 9.0042467, -0.4692738, 9.0076847, -6.8020802, 6.7278996
30: -4.1454239, 9.9664974, -4.0166926, 9.9789314, -12.7988739, 12.6729736
31: -3.1957884, 11.8157558, -3.0720024, 11.8202915, -11.5331345, 11.4037209
32: -19.0287037, -5.8217258, -19.0420074, -5.8398657, -9.3851738, 9.4449730
33: -38.5011978, -16.8358517, -38.4255104, -16.8372955, -16.0090103, 15.8998947
34: -37.8582535, -23.4196434, -37.8577957, -23.4204922, -10.3295250, 10.3128052
35: -29.0696335, -14.0528297, -29.0136204, -14.0551548, -11.8592682, 11.7809105
36: -22.0762901, -9.1922102, -22.0304375, -9.1978045, -9.2824821, 9.2204437
37: -39.7537079, -18.9863281, -39.6339035, -18.9829330, -15.8217316, 15.6984482
38: -36.0589066, -19.3190365, -36.0220833, -19.3381805, -14.5223083, 14.4625244
39: -38.4300804, -16.9097996, -38.3825226, -16.9162827, -14.8126755, 14.8033447
40: -34.3462830, -20.4506836, -34.3738594, -20.4569550, -8.6539536, 8.6933975
41: -21.2221222, -5.2890182, -21.2069893, -5.2940445, -12.3834534, 12.3960228
42: -23.4895916, -11.3867865, -23.4720955, -11.3934784, -9.8711205, 9.8622856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=163, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A2_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3366973
time: 16.51 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3366973
time: 17.19 seconds

## BFS IS instance: IS_A2_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -26.9636726, -9.4450951, -26.9906654, -9.4897499, -11.9415588, 12.0203094
1: -9.7771568, 0.0184655, -9.7960453, -0.0098925, -6.7776070, 6.8379021
2: -4.3634348, 4.8974376, -4.3716612, 4.8846245, -6.8622131, 6.9055367
3: -13.4609890, -0.5710394, -13.4910030, -0.6289885, -9.5819664, 9.6719551
4: -5.4610691, 7.3934555, -5.4994912, 7.3377113, -8.9139214, 9.0079155
5: -8.9116373, 4.2491913, -8.9413958, 4.1933742, -11.4625397, 11.5536194
6: -24.1126690, -8.9110441, -24.1219082, -8.9188280, -9.8206825, 9.8494492
7: -9.6749773, 2.7173572, -9.6943035, 2.6894646, -8.8777008, 8.9261551
8: -12.3209553, 3.1229863, -12.3427067, 3.1006913, -9.4309101, 9.4710503
9: -7.0407043, 8.6969519, -7.0883989, 8.6216631, -10.8309441, 10.9553833
10: -7.0222678, 7.2637091, -7.0660362, 7.1759953, -11.5352974, 11.6739388
11: -4.7196789, 5.0340176, -4.6724310, 5.0624156, -8.2432404, 8.1591110
12: -16.8675289, -0.5996810, -16.8740654, -0.5860680, -11.4134941, 11.4359512
13: -21.3425446, -3.0654588, -21.3794403, -3.0916090, -14.4294472, 14.5141983
14: -22.9112244, -5.0449324, -22.9060631, -5.0364027, -16.8614120, 16.8716202
15: -9.0210180, 3.5237446, -9.0554218, 3.4737513, -9.8150406, 9.9008026
16: -9.6243019, 1.1889720, -9.6532431, 1.1402421, -10.0694389, 10.1537437
17: -20.8280849, -4.1891847, -20.8143616, -4.1610813, -13.7300415, 13.6977463
18: -3.2431111, 11.7868557, -3.2244790, 11.8029699, -11.1491089, 11.1142349
19: 1.7916913, 11.0609093, 1.8471637, 11.0913410, -9.2469559, 9.1529045
20: -0.8429201, 9.8722363, -0.8054476, 9.8898048, -10.7327251, 10.6776838
21: 0.6927941, 13.0982323, 0.7505856, 13.1236916, -12.3035049, 12.2028198
22: 1.9489441, 12.2766209, 1.9842172, 12.2951422, -8.4204254, 8.3576279
23: 0.1077509, 11.0470362, 0.2059783, 11.0974827, -9.5948792, 9.4390411
24: -5.4485254, 9.4699535, -5.3659735, 9.5175648, -12.1245804, 11.9902802
25: -4.4713078, 9.7099209, -4.3895645, 9.7518129, -11.9497681, 11.8228073
26: 2.9218178, 16.2808418, 2.9805622, 16.3072739, -13.3854561, 13.3002796
27: 0.0382695, 12.1936722, 0.0928667, 12.2249546, -10.4714050, 10.3901939
28: 0.6622291, 12.6168575, 0.7340355, 12.6562071, -11.5836258, 11.4672356
29: -0.5071057, 8.9868641, -0.4695915, 9.0066528, -6.7602997, 6.7007484
30: -4.0766726, 9.9386435, -4.0172243, 9.9788122, -12.7417145, 12.6444168
31: -3.1323044, 11.7878656, -3.0733905, 11.8182163, -11.4597702, 11.3653641
32: -19.0360832, -5.8321638, -19.0488186, -5.8398924, -9.3877640, 9.4314308
33: -38.4511452, -16.8635101, -38.4260330, -16.8403893, -15.9466324, 15.8767052
34: -37.8564415, -23.4314079, -37.8628922, -23.4229660, -10.3218689, 10.3277855
35: -29.0332642, -14.0781040, -29.0141582, -14.0585203, -11.8124847, 11.7654877
36: -22.0434265, -9.2205162, -22.0322876, -9.2000790, -9.2316246, 9.2061005
37: -39.6848030, -19.0194149, -39.6345367, -18.9858513, -15.7566986, 15.6729431
38: -36.0331841, -19.3493862, -36.0269470, -19.3400040, -14.4932175, 14.4721146
39: -38.3917770, -16.9320717, -38.3833160, -16.9178963, -14.8060379, 14.7934418
40: -34.3741684, -20.4336452, -34.3860855, -20.4562607, -8.6686554, 8.7147522
41: -21.2127247, -5.3009558, -21.2106895, -5.2941227, -12.3783798, 12.3826141
42: -23.4712029, -11.4071999, -23.4714546, -11.3944664, -9.8756599, 9.8437386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3087783
time: 12.35 seconds

## Relational analysis of IS_A2_A2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A2_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3087783
time: 13.97 seconds

## BFS IS instance: IS_A2_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -26.9760628, -9.4308949, -26.9938965, -9.4893112, -12.0033798, 12.0374374
1: -9.7873363, 0.0308914, -9.7988091, -0.0095687, -6.8216820, 6.8528042
2: -4.3719110, 4.9086242, -4.3738742, 4.8848314, -6.8812943, 6.9190693
3: -13.4742804, -0.5528159, -13.4946213, -0.6287847, -9.5952339, 9.6938972
4: -5.4792056, 7.4132009, -5.5042515, 7.3379083, -8.9462166, 9.0326157
5: -8.9243431, 4.2660170, -8.9447432, 4.1935954, -11.4762344, 11.5741386
6: -24.1221294, -8.9054890, -24.1224194, -8.9176455, -9.8222771, 9.8611164
7: -9.6879559, 2.7311640, -9.6977425, 2.6895947, -8.8955994, 8.9433975
8: -12.3356590, 3.1375573, -12.3466053, 3.1008768, -9.4523907, 9.4887829
9: -7.0616140, 8.7202358, -7.0940957, 8.6218662, -10.8688393, 10.9836349
10: -7.0356030, 7.2770529, -7.0689983, 7.1764016, -11.5471115, 11.6805000
11: -4.7330952, 5.0464606, -4.6727161, 5.0655441, -8.2560844, 8.1704445
12: -16.8818226, -0.5912316, -16.8743477, -0.5839555, -11.4203377, 11.4438629
13: -21.3575001, -3.0429192, -21.3834915, -3.0910420, -14.4522285, 14.5400848
14: -22.9257164, -5.0402212, -22.9067554, -5.0353708, -16.8941574, 16.8771973
15: -9.0291519, 3.5364256, -9.0575600, 3.4742460, -9.8228798, 9.9105835
16: -9.6383877, 1.2007518, -9.6570797, 1.1405139, -10.0994110, 10.1691933
17: -20.8468361, -4.1735101, -20.8149681, -4.1569152, -13.7464218, 13.7110825
18: -3.2620516, 11.7962265, -3.2250440, 11.8053665, -11.1704521, 11.1180267
19: 1.7842748, 11.0686131, 1.8468528, 11.0934715, -9.2568474, 9.1751556
20: -0.8536673, 9.8760090, -0.8061769, 9.8907814, -10.7444487, 10.6821861
21: 0.6844056, 13.1039076, 0.7500377, 13.1251183, -12.3208847, 12.2132797
22: 1.9366760, 12.2866764, 1.9839296, 12.2979202, -8.4353523, 8.3820877
23: 0.0888848, 11.0620813, 0.2056563, 11.1015329, -9.6176949, 9.4620476
24: -5.4693680, 9.4890699, -5.3662882, 9.5227776, -12.1505127, 12.0086823
25: -4.4920640, 9.7285175, -4.3899336, 9.7567635, -11.9755096, 11.8470535
26: 2.9033499, 16.2907867, 2.9797940, 16.3098431, -13.4064932, 13.3109932
27: 0.0235806, 12.2027321, 0.0925760, 12.2274084, -10.4851990, 10.3954811
28: 0.6468964, 12.6304789, 0.7335052, 12.6598063, -11.6023865, 11.4970856
29: -0.5205663, 9.0014439, -0.4696891, 9.0102272, -6.7773209, 6.7244568
30: -4.0975423, 9.9561634, -4.0176535, 9.9835920, -12.7672195, 12.6704712
31: -3.1483521, 11.8005209, -3.0736642, 11.8216734, -11.4790192, 11.3908005
32: -19.0416050, -5.8309436, -19.0492420, -5.8395476, -9.3953705, 9.4407616
33: -38.4623642, -16.8544807, -38.4266624, -16.8384514, -15.9579239, 15.8867760
34: -37.8709869, -23.4212227, -37.8632736, -23.4202232, -10.3315849, 10.3183289
35: -29.0405712, -14.0693312, -29.0145721, -14.0567808, -11.8212433, 11.7738457
36: -22.0455055, -9.2172489, -22.0323944, -9.1992188, -9.2554588, 9.2122517
37: -39.7003326, -19.0075245, -39.6349487, -18.9829597, -15.7652206, 15.6822052
38: -36.0367737, -19.3489838, -36.0266953, -19.3395576, -14.4825211, 14.4726181
39: -38.3947716, -16.9322739, -38.3831100, -16.9178047, -14.7762375, 14.7924881
40: -34.3780823, -20.4341793, -34.3866043, -20.4565372, -8.6760330, 8.7177200
41: -21.2185516, -5.2952385, -21.2113152, -5.2931280, -12.3846359, 12.3972511
42: -23.4798164, -11.3993225, -23.4718285, -11.3927593, -9.8713913, 9.8508759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3184017
time: 12.84 seconds

## Relational analysis of IS_A2_A2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3184017
time: 12.94 seconds

## BFS IS instance: IS_A2_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -26.9725342, -9.4201708, -26.9907494, -9.4900570, -11.9482155, 12.0494804
1: -9.7833691, 0.0355721, -9.7960749, -0.0101223, -6.7834110, 6.8562737
2: -4.3667822, 4.9027042, -4.3715634, 4.8844814, -6.8620110, 6.9198532
3: -13.4734592, -0.5372667, -13.4913301, -0.6291203, -9.5941467, 9.7061043
4: -5.4767900, 7.4205289, -5.4996128, 7.3376713, -8.9296188, 9.0349808
5: -8.9259720, 4.2807856, -8.9415874, 4.1933804, -11.4759445, 11.5854225
6: -24.1159592, -8.9082422, -24.1210880, -8.9190006, -9.8251038, 9.8522301
7: -9.6818638, 2.7278557, -9.6942120, 2.6893005, -8.8848190, 8.9366493
8: -12.3283892, 3.1317120, -12.3426094, 3.1005645, -9.4370689, 9.4809647
9: -7.0543509, 8.7352524, -7.0887203, 8.6215324, -10.8449669, 10.9938850
10: -7.0374722, 7.3023968, -7.0664759, 7.1759062, -11.5501862, 11.7139854
11: -4.7629957, 5.0515966, -4.6723781, 5.0627861, -8.2865295, 8.1763077
12: -16.8765602, -0.5679882, -16.8736572, -0.5860860, -11.4206848, 11.4688072
13: -21.3719330, -2.9884133, -21.3802147, -3.0917282, -14.4581261, 14.5915031
14: -22.9284248, -5.0212307, -22.9059410, -5.0365887, -16.8786545, 16.8953018
15: -9.0385208, 3.5570884, -9.0553513, 3.4736347, -9.8329964, 9.9356956
16: -9.6290655, 1.1964664, -9.6532507, 1.1398778, -10.0695114, 10.1744385
17: -20.8415375, -4.1658192, -20.8140602, -4.1612091, -13.7445068, 13.7189484
18: -3.2757111, 11.7975388, -3.2242541, 11.8031836, -11.1819916, 11.1246719
19: 1.7496722, 11.0753517, 1.8472631, 11.0918388, -9.2892685, 9.1670914
20: -0.8768499, 9.8830442, -0.8052614, 9.8902025, -10.7670527, 10.6883059
21: 0.6380162, 13.1161804, 0.7508490, 13.1242390, -12.3587265, 12.2203026
22: 1.9299421, 12.2825546, 1.9845753, 12.2952385, -8.4453087, 8.3611374
23: 0.0469372, 11.0702372, 0.2059811, 11.0982084, -9.6561127, 9.4621468
24: -5.5094380, 9.4915562, -5.3658895, 9.5182905, -12.1859589, 12.0118942
25: -4.5076509, 9.7258968, -4.3894444, 9.7521486, -11.9863472, 11.8389587
26: 2.8693452, 16.2987404, 2.9807673, 16.3078041, -13.4384594, 13.3179731
27: -0.0136387, 12.2111616, 0.0931022, 12.2256222, -10.5240440, 10.4075241
28: 0.6131928, 12.6365824, 0.7340817, 12.6567602, -11.6329498, 11.4868393
29: -0.5309244, 8.9960041, -0.4693776, 9.0068436, -6.7843647, 6.7099724
30: -4.1327333, 9.9601965, -4.0171261, 9.9794912, -12.7978745, 12.6662369
31: -3.1780767, 11.8018618, -3.0731559, 11.8185339, -11.5054703, 11.3791962
32: -19.0449028, -5.8106365, -19.0489349, -5.8399415, -9.3958893, 9.4545403
33: -38.4731789, -16.8536530, -38.4259605, -16.8402767, -15.9814072, 15.8832626
34: -37.8574448, -23.4262028, -37.8618927, -23.4230309, -10.3309402, 10.3293076
35: -29.0438461, -14.0723877, -29.0141029, -14.0585136, -11.8399963, 11.7666245
36: -22.0500259, -9.2082233, -22.0308247, -9.2000484, -9.2401695, 9.2075443
37: -39.7216568, -19.0039387, -39.6344147, -18.9856319, -15.7950516, 15.6851120
38: -36.0423660, -19.3259182, -36.0245590, -19.3400688, -14.5162964, 14.4653549
39: -38.4061050, -16.9203091, -38.3834610, -16.9179344, -14.8211365, 14.8029633
40: -34.3771667, -20.4313717, -34.3858566, -20.4565506, -8.6712875, 8.7166042
41: -21.2280121, -5.2908697, -21.2105427, -5.2947493, -12.3888779, 12.3938904
42: -23.4793510, -11.3965702, -23.4712410, -11.3946342, -9.8836212, 9.8539619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A1_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3354427
time: 13.94 seconds

## Relational analysis of IS_A2_A2_A1_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3354427
time: 14.31 seconds

## BFS IS instance: IS_A2_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -26.9849243, -9.4060011, -26.9939671, -9.4896355, -12.0100441, 12.0666237
1: -9.7935371, 0.0480108, -9.7988672, -0.0097656, -6.8274899, 6.8711739
2: -4.3752894, 4.9138970, -4.3737764, 4.8846507, -6.8811073, 6.9333801
3: -13.4867964, -0.5190151, -13.4949570, -0.6289296, -9.6074028, 9.7280502
4: -5.4949193, 7.4402742, -5.5043697, 7.3378372, -8.9619141, 9.0596657
5: -8.9386845, 4.2976146, -8.9449368, 4.1935954, -11.4896545, 11.6059265
6: -24.1254005, -8.9027195, -24.1216087, -8.9178133, -9.8267059, 9.8638821
7: -9.6948309, 2.7416167, -9.6976824, 2.6894336, -8.9027290, 8.9539070
8: -12.3430519, 3.1463397, -12.3464985, 3.1007664, -9.4585381, 9.4987087
9: -7.0752559, 8.7584925, -7.0944252, 8.6217613, -10.8828278, 11.0221329
10: -7.0507364, 7.3157711, -7.0694146, 7.1763163, -11.5620079, 11.7205505
11: -4.7764292, 5.0640240, -4.6726799, 5.0658784, -8.2993889, 8.1876411
12: -16.8908958, -0.5595726, -16.8739090, -0.5839981, -11.4275208, 11.4767303
13: -21.3869095, -2.9658632, -21.3842716, -3.0912313, -14.4808998, 14.6173706
14: -22.9428616, -5.0165310, -22.9065857, -5.0355358, -16.9114075, 16.9008484
15: -9.0466805, 3.5697703, -9.0574741, 3.4741311, -9.8408318, 9.9454994
16: -9.6431332, 1.2082434, -9.6570606, 1.1401615, -10.0995140, 10.1898994
17: -20.8603115, -4.1501226, -20.8147202, -4.1570330, -13.7609177, 13.7323151
18: -3.2946372, 11.8068762, -3.2248054, 11.8055983, -11.2033691, 11.1284637
19: 1.7422392, 11.0830660, 1.8469419, 11.0939617, -9.2991791, 9.1893349
20: -0.8875995, 9.8868122, -0.8059874, 9.8911791, -10.7787781, 10.6927996
21: 0.6296468, 13.1218643, 0.7503123, 13.1256971, -12.3760834, 12.2307854
22: 1.9176884, 12.2926140, 1.9842987, 12.2980242, -8.4602661, 8.3855820
23: 0.0280766, 11.0852680, 0.2056574, 11.1022596, -9.6789436, 9.4851646
24: -5.5302405, 9.5106640, -5.3662624, 9.5234823, -12.2118912, 12.0303459
25: -4.5284071, 9.7444839, -4.3898020, 9.7570400, -12.0120697, 11.8632011
26: 2.8508630, 16.3087063, 2.9800453, 16.3103752, -13.4595127, 13.3286610
27: -0.0283213, 12.2201691, 0.0928359, 12.2280540, -10.5378494, 10.4128189
28: 0.5978341, 12.6501989, 0.7335553, 12.6603527, -11.6517181, 11.5167313
29: -0.5444226, 9.0105648, -0.4695016, 9.0104113, -6.8013916, 6.7336960
30: -4.1535807, 9.9776859, -4.0175743, 9.9842882, -12.8234100, 12.6922684
31: -3.1941025, 11.8145533, -3.0734406, 11.8219852, -11.5246964, 11.4046707
32: -19.0503998, -5.8094339, -19.0493431, -5.8395782, -9.4034843, 9.4638786
33: -38.4844055, -16.8445778, -38.4265289, -16.8383064, -15.9926987, 15.8933563
34: -37.8720245, -23.4160442, -37.8622589, -23.4202881, -10.3406868, 10.3198853
35: -29.0511475, -14.0636511, -29.0145283, -14.0567894, -11.8487549, 11.7749825
36: -22.0521297, -9.2049389, -22.0309391, -9.1992178, -9.2639771, 9.2136784
37: -39.7372513, -18.9920864, -39.6348763, -18.9827404, -15.8035660, 15.6943283
38: -36.0459442, -19.3254681, -36.0243225, -19.3395691, -14.5055923, 14.4658661
39: -38.4091644, -16.9205933, -38.3833008, -16.9179001, -14.7913513, 14.8019867
40: -34.3811111, -20.4318886, -34.3863983, -20.4568100, -8.6786575, 8.7195454
41: -21.2338142, -5.2851415, -21.2111778, -5.2937584, -12.3951340, 12.4085350
42: -23.4879532, -11.3886776, -23.4716320, -11.3929005, -9.8793411, 9.8610916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A1_A2_A2_B1

### Relational analysis result of IS_A2_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3450677
time: 15.08 seconds

## Relational analysis of IS_A2_A2_A1_A2_A2_B2

### Relational analysis result of IS_A2_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3450677
time: 16.18 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -26.9713707, -9.4189510, -26.9907780, -9.4895821, -11.9508858, 12.0537643
1: -9.7803221, 0.0356293, -9.7955294, -0.0097535, -6.7860680, 6.8611069
2: -4.3656683, 4.9049826, -4.3717155, 4.8847017, -6.8652916, 6.9145660
3: -13.4714088, -0.5496447, -13.4926491, -0.6289546, -9.5915871, 9.6954041
4: -5.4787011, 7.4201117, -5.5022602, 7.3376312, -8.9296837, 9.0358124
5: -8.9241714, 4.2692032, -8.9433270, 4.1934342, -11.4739075, 11.5756149
6: -24.1052437, -8.9110222, -24.1185989, -8.9185448, -9.8176231, 9.8486919
7: -9.6832008, 2.7367716, -9.6953220, 2.6895533, -8.8841171, 8.9474068
8: -12.3301392, 3.1356745, -12.3432388, 3.1005960, -9.4364014, 9.4755650
9: -7.0639639, 8.7342196, -7.0920343, 8.6216764, -10.8524513, 10.9962082
10: -7.0504866, 7.3081617, -7.0705409, 7.1758242, -11.5606079, 11.7227745
11: -4.7298255, 5.0434580, -4.6724606, 5.0636029, -8.2546196, 8.1685753
12: -16.8849945, -0.5917953, -16.8739910, -0.5850275, -11.4319725, 11.4415588
13: -21.3611794, -3.0560503, -21.3793564, -3.0913401, -14.4313354, 14.5277977
14: -22.9205685, -5.0366030, -22.9062557, -5.0355864, -16.8649826, 16.8821869
15: -9.0441875, 3.5487967, -9.0583553, 3.4736719, -9.8382339, 9.9289131
16: -9.6392803, 1.2303586, -9.6556845, 1.1404643, -10.0817299, 10.1981888
17: -20.8608150, -4.1680789, -20.8146248, -4.1578846, -13.7653275, 13.7165604
18: -3.2476153, 11.7950144, -3.2248404, 11.8030319, -11.1530304, 11.1229935
19: 1.7750585, 11.0716610, 1.8472390, 11.0930443, -9.2680931, 9.1630707
20: -0.8524828, 9.8735580, -0.8056843, 9.8893328, -10.7418156, 10.6792421
21: 0.6860664, 13.0987043, 0.7503848, 13.1221905, -12.3281021, 12.2135201
22: 1.9301157, 12.2822666, 1.9839945, 12.2958422, -8.4418488, 8.3642464
23: 0.0774987, 11.0662117, 0.2061599, 11.1005955, -9.6283798, 9.4543686
24: -5.4701905, 9.4846754, -5.3660617, 9.5197105, -12.1478806, 12.0042114
25: -4.5034661, 9.7292461, -4.3896618, 9.7548895, -11.9850502, 11.8403282
26: 2.9025912, 16.2801895, 2.9800529, 16.3059044, -13.4033127, 13.3001366
27: 0.0300848, 12.2007761, 0.0927997, 12.2257042, -10.4738770, 10.3955307
28: 0.6394958, 12.6330214, 0.7340393, 12.6586170, -11.6077271, 11.4820099
29: -0.5234414, 8.9937572, -0.4697293, 9.0076790, -6.7780685, 6.7073956
30: -4.0894370, 9.9483519, -4.0173831, 9.9798937, -12.7460556, 12.6475067
31: -3.1473727, 11.8006077, -3.0731759, 11.8194618, -11.4850616, 11.3774109
32: -19.0349998, -5.8294253, -19.0474396, -5.8396001, -9.3872757, 9.4330788
33: -38.4742775, -16.8477669, -38.4261627, -16.8380871, -15.9726257, 15.8919449
34: -37.8568954, -23.4231834, -37.8624878, -23.4221306, -10.3231049, 10.3374252
35: -29.0549583, -14.0610619, -29.0140953, -14.0558786, -11.8278961, 11.7791138
36: -22.0737038, -9.2018738, -22.0323257, -9.1970692, -9.2560806, 9.2216892
37: -39.7159271, -19.0030212, -39.6345329, -18.9832649, -15.7901535, 15.6878662
38: -36.0560837, -19.3328743, -36.0270081, -19.3375359, -14.5188141, 14.4878273
39: -38.4173279, -16.9171867, -38.3834839, -16.9155655, -14.8333893, 14.8068695
40: -34.3651161, -20.4342728, -34.3816414, -20.4561691, -8.6646538, 8.7156849
41: -21.2131100, -5.2933474, -21.2100925, -5.2932920, -12.3797760, 12.3880615
42: -23.4736271, -11.4027863, -23.4713745, -11.3941336, -9.8797836, 9.8480759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A2_A1_A1_A1

### Relational analysis result of IS_A2_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3052956
time: 12.80 seconds

## Relational analysis of IS_A2_A2_A2_A1_A1_A2

### Relational analysis result of IS_A2_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3080941
time: 10.15 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -26.9837990, -9.4047861, -26.9940186, -9.4891291, -12.0127029, 12.0709000
1: -9.7904987, 0.0480795, -9.7983093, -0.0094199, -6.8301430, 6.8760147
2: -4.3741398, 4.9161692, -4.3739347, 4.8849001, -6.8843689, 6.9280777
3: -13.4847488, -0.5314269, -13.4962654, -0.6287365, -9.6048393, 9.7173462
4: -5.4968567, 7.4398613, -5.5070133, 7.3378057, -8.9619904, 9.0604973
5: -8.9368877, 4.2860389, -8.9466724, 4.1935978, -11.4876404, 11.5961342
6: -24.1146832, -8.9054909, -24.1191006, -8.9173660, -9.8192101, 9.8603516
7: -9.6961699, 2.7505403, -9.6987686, 2.6896610, -8.9020157, 8.9646759
8: -12.3448229, 3.1502721, -12.3471775, 3.1008234, -9.4578819, 9.4933224
9: -7.0848613, 8.7574434, -7.0977492, 8.6219168, -10.8903313, 11.0244484
10: -7.0638328, 7.3215303, -7.0734921, 7.1762142, -11.5724106, 11.7293396
11: -4.7432485, 5.0558991, -4.6727390, 5.0667038, -8.2674713, 8.1798706
12: -16.8992805, -0.5834018, -16.8743095, -0.5828698, -11.4388390, 11.4494972
13: -21.3761597, -3.0334911, -21.3834457, -3.0907965, -14.4540787, 14.5537071
14: -22.9349747, -5.0318918, -22.9069748, -5.0345201, -16.8976822, 16.8877716
15: -9.0523357, 3.5614753, -9.0604887, 3.4741826, -9.8460732, 9.9387093
16: -9.6533670, 1.2421541, -9.6594973, 1.1407351, -10.1117096, 10.2136612
17: -20.8795872, -4.1524062, -20.8152580, -4.1537151, -13.7816925, 13.7298889
18: -3.2665308, 11.8043251, -3.2253675, 11.8054752, -11.1743660, 11.1267624
19: 1.7676299, 11.0793743, 1.8469276, 11.0951710, -9.2779732, 9.1853142
20: -0.8632228, 9.8773317, -0.8063884, 9.8903236, -10.7535467, 10.6837196
21: 0.6776609, 13.1043806, 0.7498360, 13.1236401, -12.3454590, 12.2239914
22: 1.9178457, 12.2923288, 1.9836922, 12.2986422, -8.4567986, 8.3886986
23: 0.0586424, 11.0812368, 0.2058529, 11.1046677, -9.6512299, 9.4773788
24: -5.4910450, 9.5038013, -5.3663554, 9.5248871, -12.1738205, 12.0226364
25: -4.5242233, 9.7478418, -4.3900094, 9.7597733, -12.0107689, 11.8645821
26: 2.8841248, 16.2901344, 2.9792781, 16.3084793, -13.4243546, 13.3108559
27: 0.0153983, 12.2097435, 0.0925286, 12.2281227, -10.4877014, 10.4008102
28: 0.6241701, 12.6466074, 0.7335095, 12.6622257, -11.6265182, 11.5118790
29: -0.5369362, 9.0083265, -0.4698392, 9.0112247, -6.7951012, 6.7311058
30: -4.1102729, 9.9658537, -4.0177898, 9.9846973, -12.7715912, 12.6735535
31: -3.1634123, 11.8132868, -3.0734501, 11.8229008, -11.5043221, 11.4028358
32: -19.0405102, -5.8281860, -19.0478439, -5.8392553, -9.3948746, 9.4424324
33: -38.4855042, -16.8387032, -38.4267731, -16.8361282, -15.9839401, 15.9020462
34: -37.8714523, -23.4130440, -37.8628769, -23.4193687, -10.3328590, 10.3279419
35: -29.0622578, -14.0522976, -29.0145283, -14.0541744, -11.8366623, 11.7874565
36: -22.0757828, -9.1985979, -22.0324287, -9.1962509, -9.2798309, 9.2278252
37: -39.7315369, -18.9911385, -39.6349335, -18.9803581, -15.7986755, 15.6971436
38: -36.0596695, -19.3324509, -36.0268173, -19.3370590, -14.5080566, 14.4883347
39: -38.4203644, -16.9174709, -38.3832474, -16.9154739, -14.8036041, 14.8059464
40: -34.3690872, -20.4348106, -34.3821907, -20.4564304, -8.6720276, 8.7186661
41: -21.2189560, -5.2875948, -21.2107048, -5.2923002, -12.3860168, 12.4027023
42: -23.4822464, -11.3949251, -23.4717903, -11.3924160, -9.8754845, 9.8552055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A2_A1_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3149200
time: 31.23 seconds

## Relational analysis of IS_A2_A2_A2_A1_A2_A2

### Relational analysis result of IS_A2_A2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3177180
time: 27.83 seconds

## BFS IS instance: IS_A2_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -26.9802780, -9.3940859, -26.9908638, -9.4898815, -11.9575424, 12.0829048
1: -9.7865353, 0.0527253, -9.7955914, -0.0099611, -6.7918644, 6.8794594
2: -4.3690443, 4.9102311, -4.3716106, 4.8845448, -6.8651085, 6.9288845
3: -13.4838905, -0.5158687, -13.4929504, -0.6291182, -9.6037483, 9.7295418
4: -5.4944263, 7.4471560, -5.5023494, 7.3375702, -8.9453964, 9.0628395
5: -8.9385071, 4.3008118, -8.9435329, 4.1933908, -11.4873047, 11.6074219
6: -24.1084995, -8.9082336, -24.1177807, -8.9187050, -9.8220406, 9.8514423
7: -9.6900826, 2.7472124, -9.6952295, 2.6893625, -8.8912621, 8.9579124
8: -12.3376083, 3.1444328, -12.3431244, 3.1004810, -9.4425812, 9.4854946
9: -7.0776348, 8.7725229, -7.0923667, 8.6215639, -10.8664627, 11.0347137
10: -7.0656576, 7.3468485, -7.0709510, 7.1757183, -11.5754890, 11.7628174
11: -4.7731433, 5.0610347, -4.6723924, 5.0639482, -8.2979126, 8.1857758
12: -16.8940048, -0.5601028, -16.8736305, -0.5850089, -11.4391479, 11.4744415
13: -21.3905716, -2.9789972, -21.3801880, -3.0914660, -14.4600487, 14.6050911
14: -22.9376755, -5.0128908, -22.9061356, -5.0357647, -16.8822098, 16.9058838
15: -9.0617371, 3.5821443, -9.0582905, 3.4735556, -9.8562126, 9.9638138
16: -9.6440268, 1.2378473, -9.6556616, 1.1401043, -10.0818329, 10.2188683
17: -20.8742676, -4.1446829, -20.8143539, -4.1580038, -13.7798080, 13.7377930
18: -3.2802069, 11.8056507, -3.2246189, 11.8032789, -11.1859093, 11.1334343
19: 1.7330396, 11.0861034, 1.8473456, 11.0935469, -9.3104019, 9.1772423
20: -0.8863750, 9.8843594, -0.8054588, 9.8897457, -10.7761211, 10.6898184
21: 0.6312995, 13.1166821, 0.7506673, 13.1227474, -12.3833160, 12.2309990
22: 1.9111300, 12.2882166, 1.9843488, 12.2959681, -8.4667244, 8.3677597
23: 0.0166934, 11.0894108, 0.2061771, 11.1013060, -9.6896248, 9.4774780
24: -5.5311298, 9.5062733, -5.3659301, 9.5204239, -12.2092514, 12.0258179
25: -4.5398207, 9.7452326, -4.3895321, 9.7551823, -12.0216179, 11.8564606
26: 2.8501353, 16.2980995, 2.9802608, 16.3064423, -13.4563065, 13.3178387
27: -0.0217953, 12.2182121, 0.0930336, 12.2263432, -10.5265121, 10.4128265
28: 0.5904529, 12.6527119, 0.7340887, 12.6591578, -11.6570663, 11.5016403
29: -0.5472610, 9.0028744, -0.4695387, 9.0078583, -6.8021107, 6.7166252
30: -4.1454749, 9.9698496, -4.0172472, 9.9805841, -12.8022690, 12.6693497
31: -3.1931396, 11.8146000, -3.0729616, 11.8197861, -11.5307388, 11.3912697
32: -19.0438156, -5.8078794, -19.0475407, -5.8396244, -9.3954086, 9.4561920
33: -38.4963455, -16.8378716, -38.4260635, -16.8379555, -16.0074005, 15.8985367
34: -37.8578949, -23.4179745, -37.8615112, -23.4222069, -10.3321915, 10.3389435
35: -29.0655556, -14.0553894, -29.0140228, -14.0558949, -11.8554153, 11.7802353
36: -22.0803432, -9.1896029, -22.0308838, -9.1970673, -9.2646141, 9.2231293
37: -39.7528191, -18.9875679, -39.6344070, -18.9830627, -15.8285370, 15.7000198
38: -36.0652542, -19.3093834, -36.0246429, -19.3375626, -14.5418472, 14.4810600
39: -38.4316788, -16.9054222, -38.3836098, -16.9156342, -14.8485107, 14.8164062
40: -34.3681793, -20.4320107, -34.3814468, -20.4564438, -8.6672821, 8.7175407
41: -21.2283745, -5.2832336, -21.2099590, -5.2939262, -12.3902435, 12.3993263
42: -23.4817505, -11.3921509, -23.4711323, -11.3942747, -9.8877487, 9.8583069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443855, upper bound: 6.3319599
time: 15.04 seconds

## Relational analysis of IS_A2_A2_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443855, upper bound: 6.3347587
time: 12.02 seconds

## BFS IS instance: IS_A2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -26.9926434, -9.3798542, -26.9940968, -9.4894953, -12.0193481, 12.1000557
1: -9.7966948, 0.0651484, -9.7983589, -0.0096221, -6.8359432, 6.8943634
2: -4.3775377, 4.9214058, -4.3738370, 4.8847227, -6.8841820, 6.9423904
3: -13.4972363, -0.4976311, -13.4965687, -0.6288981, -9.6169968, 9.7514954
4: -5.5125852, 7.4669089, -5.5071368, 7.3377423, -8.9776917, 9.0875435
5: -8.9512272, 4.3176003, -8.9468727, 4.1936035, -11.5009995, 11.6279297
6: -24.1179504, -8.9027004, -24.1183128, -8.9175205, -9.8236237, 9.8631172
7: -9.7030563, 2.7610111, -9.6986694, 2.6895061, -8.9091682, 8.9751854
8: -12.3522711, 3.1590157, -12.3470535, 3.1006675, -9.4640541, 9.5032349
9: -7.0985422, 8.7957478, -7.0980730, 8.6217995, -10.9043388, 11.0629425
10: -7.0789914, 7.3602238, -7.0739174, 7.1761165, -11.5872917, 11.7693787
11: -4.7865410, 5.0734673, -4.6727080, 5.0670605, -8.3107529, 8.1971054
12: -16.9083004, -0.5517282, -16.8738880, -0.5829015, -11.4460335, 11.4823875
13: -21.4055557, -2.9564905, -21.3842697, -3.0909605, -14.4827652, 14.6309776
14: -22.9521523, -5.0081663, -22.9068127, -5.0347023, -16.9149399, 16.9114456
15: -9.0699005, 3.5948188, -9.0604086, 3.4740648, -9.8640556, 9.9736023
16: -9.6581173, 1.2496309, -9.6594791, 1.1403875, -10.1118393, 10.2343369
17: -20.8930817, -4.1289988, -20.8150387, -4.1538391, -13.7961807, 13.7511368
18: -3.2991290, 11.8150053, -3.2251918, 11.8056688, -11.2072830, 11.1372261
19: 1.7256103, 11.0938168, 1.8470259, 11.0956736, -9.3202896, 9.1994781
20: -0.8971062, 9.8881435, -0.8061938, 9.8907375, -10.7878437, 10.6943378
21: 0.6229107, 13.1223450, 0.7501178, 13.1241817, -12.4006653, 12.2414322
22: 1.8988733, 12.2982750, 1.9840431, 12.2987385, -8.4816895, 8.3922100
23: -0.0021305, 11.1044788, 0.2058495, 11.1053782, -9.7124634, 9.5004921
24: -5.5519290, 9.5253735, -5.3663015, 9.5256147, -12.2351837, 12.0442505
25: -4.5605597, 9.7638178, -4.3898883, 9.7600794, -12.0473099, 11.8807487
26: 2.8316593, 16.3080521, 2.9795346, 16.3090115, -13.4773521, 13.3285179
27: -0.0364749, 12.2272472, 0.0927784, 12.2287827, -10.5403252, 10.4181404
28: 0.5751112, 12.6663227, 0.7335415, 12.6627712, -11.6758423, 11.5314865
29: -0.5607300, 9.0174541, -0.4696536, 9.0114250, -6.8191338, 6.7403355
30: -4.1663566, 9.9873867, -4.0176992, 9.9853840, -12.8277893, 12.6953812
31: -3.2091265, 11.8273010, -3.0732203, 11.8232374, -11.5499802, 11.4166946
32: -19.0493469, -5.8066530, -19.0479469, -5.8392959, -9.4030113, 9.4655304
33: -38.5075531, -16.8287773, -38.4266357, -16.8359833, -16.0186768, 15.9086304
34: -37.8724480, -23.4078445, -37.8618851, -23.4194489, -10.3419380, 10.3294754
35: -29.0728283, -14.0466156, -29.0144444, -14.0541430, -11.8641510, 11.7885780
36: -22.0824375, -9.1863241, -22.0309696, -9.1962423, -9.2883453, 9.2292461
37: -39.7684174, -18.9757233, -39.6348419, -18.9801693, -15.8370132, 15.7092743
38: -36.0688477, -19.3089905, -36.0243797, -19.3371067, -14.5311279, 14.4815826
39: -38.4347115, -16.9056664, -38.3833847, -16.9155121, -14.8187332, 14.8154602
40: -34.3720703, -20.4325409, -34.3819695, -20.4567184, -8.6746521, 8.7205067
41: -21.2341881, -5.2774973, -21.2105675, -5.2929235, -12.3965073, 12.4139938
42: -23.4903755, -11.3842678, -23.4715462, -11.3925743, -9.8834457, 9.8654442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=20, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443855, upper bound: 6.3415859
time: 11.10 seconds

## Relational analysis of IS_A2_A2_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3443855, upper bound: 6.3443843
time: 14.02 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 27.34 seconds
IS_A1_A2_A1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3010856
IS_A1_A2_A1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3010856
IS_A1_A2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3107124
IS_A1_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3107124
IS_A1_A2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3277490
IS_A1_A2_A1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3277490
IS_A1_A2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3373769
IS_A1_A2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3373769
IS_A1_A2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3004058
IS_A1_A2_A2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3004058
IS_A1_A2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3100329
IS_A1_A2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3100329
IS_A1_A2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3270691
IS_A1_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443854, upper bound: 6.3270691
IS_A1_A2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3366973
IS_A1_A2_A2_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3366973
IS_A2_A2_A1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3087783
IS_A2_A2_A1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3087783
IS_A2_A2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3184017
IS_A2_A2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415032, upper bound: 6.3184017
IS_A2_A2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3354427
IS_A2_A2_A1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3354427
IS_A2_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3450677
IS_A2_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3415870, upper bound: 6.3450677
IS_A2_A2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3052956
IS_A2_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3080941
IS_A2_A2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3149200
IS_A2_A2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443016, upper bound: 6.3177180
IS_A2_A2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443855, upper bound: 6.3319599
IS_A2_A2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443855, upper bound: 6.3347587
IS_A2_A2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443855, upper bound: 6.3415859
IS_A2_A2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 27.34
Output dim: 26, lower bound: -6.3443855, upper bound: 6.3443843

## BFS IS instance: IS_A1_A2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -26.9658318, -9.4468498, -26.9900646, -9.4720583, -12.0116882, 12.0161171
1: -9.7812910, 0.0188744, -9.7959023, -0.0018649, -6.8257980, 6.8333626
2: -4.3682523, 4.9043808, -4.3729906, 4.8891263, -6.8821220, 6.9058151
3: -13.4603338, -0.5710816, -13.4900808, -0.6118944, -9.6010513, 9.6678352
4: -5.4565978, 7.3949280, -5.4969292, 7.3540740, -8.9402313, 9.0036163
5: -8.9074259, 4.2497501, -8.9395914, 4.2083125, -11.4757462, 11.5469093
6: -24.0946617, -8.9244452, -24.1115074, -8.9185696, -9.8020744, 9.8294983
7: -9.6795158, 2.7208319, -9.6952744, 2.7000427, -8.8978691, 8.9270401
8: -12.3294277, 3.1321917, -12.3454018, 3.1099632, -9.4528656, 9.4797058
9: -7.0465088, 8.7041521, -7.0892916, 8.6454792, -10.8786430, 10.9579544
10: -7.0095186, 7.2496433, -7.0611286, 7.2060914, -11.5549164, 11.6368256
11: -4.7221808, 5.0353847, -4.6797380, 5.0632458, -8.2370987, 8.1669426
12: -16.8756752, -0.5973518, -16.8845959, -0.5854919, -11.4084015, 11.4385910
13: -21.3538055, -3.0526967, -21.3918667, -3.0927453, -14.4499359, 14.5187950
14: -22.9038105, -5.0611420, -22.9094810, -5.0420980, -16.8643723, 16.8538971
15: -9.0135946, 3.5196452, -9.0551701, 3.4902892, -9.8244667, 9.8881912
16: -9.6255836, 1.1857929, -9.6519756, 1.1623440, -10.1101494, 10.1473045
17: -20.8282585, -4.1932964, -20.8324165, -4.1633544, -13.7134399, 13.7083435
18: -3.2519312, 11.7893219, -3.2224956, 11.8045673, -11.1595230, 11.1051826
19: 1.7997341, 11.0551929, 1.8382752, 11.0890760, -9.2354012, 9.1720848
20: -0.8371906, 9.8686552, -0.8113678, 9.8880348, -10.7252254, 10.6800232
21: 0.7014298, 13.0907383, 0.7451458, 13.1209888, -12.2970352, 12.2096214
22: 1.9517303, 12.2761011, 1.9742565, 12.2942886, -8.4143639, 8.3820782
23: 0.1197908, 11.0281572, 0.1873029, 11.0899935, -9.5721283, 9.4519081
24: -5.4457922, 9.4651585, -5.3808255, 9.5150957, -12.1175003, 12.0033798
25: -4.4642930, 9.6970997, -4.4072275, 9.7463999, -11.9341507, 11.8351479
26: 2.9306507, 16.2767925, 2.9688063, 16.3046207, -13.3739700, 13.3079863
27: 0.0409484, 12.1898212, 0.0832124, 12.2229042, -10.4653244, 10.3900795
28: 0.6732528, 12.6032028, 0.7190788, 12.6516018, -11.5661926, 11.4862061
29: -0.5076499, 8.9879742, -0.4791844, 9.0062580, -6.7588997, 6.7218227
30: -4.0764418, 9.9349661, -4.0293641, 9.9774666, -12.7375259, 12.6578979
31: -3.1348648, 11.7887516, -3.0804498, 11.8202496, -11.4615784, 11.3877411
32: -19.0198612, -5.8461676, -19.0421295, -5.8391895, -9.3783150, 9.4179955
33: -38.4558105, -16.8617783, -38.4328499, -16.8395557, -15.9479065, 15.8866348
34: -37.8561859, -23.4331512, -37.8582001, -23.4195137, -10.3234863, 10.2981186
35: -29.0371628, -14.0759258, -29.0216389, -14.0579243, -11.8152733, 11.7722473
36: -22.0392532, -9.2234812, -22.0482788, -9.2013941, -9.2472572, 9.2110767
37: -39.6854668, -19.0185432, -39.6498413, -18.9860916, -15.7470245, 15.6863174
38: -36.0266380, -19.3594360, -36.0348587, -19.3410263, -14.4717102, 14.4601746
39: -38.3899765, -16.9368114, -38.3916588, -16.9192505, -14.7676620, 14.7856979
40: -34.3514900, -20.4523735, -34.3772964, -20.4455109, -8.6667557, 8.6852989
41: -21.2060642, -5.3072166, -21.2076702, -5.2947102, -12.3710022, 12.3782616
42: -23.4777565, -11.4020624, -23.4705296, -11.3942184, -9.8546753, 9.8502502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=64, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=163, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A1_A2_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3412807, upper bound: 6.2747062
time: 22.65 seconds

## Relational analysis of IS_A1_A2_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_A1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3412807, upper bound: 6.3104903
time: 24.41 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -26.9702911, -9.4099827, -26.9870930, -9.4726391, -11.9659767, 12.0616264
1: -9.7807255, 0.0406976, -9.7930088, -0.0022683, -6.7959805, 6.8599930
2: -4.3654494, 4.9059620, -4.3707442, 4.8888283, -6.8659897, 6.9156647
3: -13.4699469, -0.5341387, -13.4884262, -0.6122208, -9.6095886, 9.7034721
4: -5.4718390, 7.4288950, -5.4950342, 7.3537726, -8.9394112, 9.0338516
5: -8.9216213, 4.2845631, -8.9383831, 4.2081447, -11.4867477, 11.5802116
6: -24.0811825, -8.9271936, -24.1070251, -8.9196568, -9.8018456, 9.8198280
7: -9.6816940, 2.7368970, -9.6927443, 2.6998267, -8.8934631, 8.9415703
8: -12.3315220, 3.1390748, -12.3419733, 3.1095095, -9.4431400, 9.4763470
9: -7.0625648, 8.7564135, -7.0875492, 8.6451435, -10.8762779, 11.0090370
10: -7.0396061, 7.3194623, -7.0631008, 7.2054119, -11.5833168, 11.7191467
11: -4.7622313, 5.0500956, -4.6793962, 5.0617290, -8.2789040, 8.1822777
12: -16.8878593, -0.5662293, -16.8838406, -0.5865042, -11.4272690, 11.4689178
13: -21.3868484, -2.9886413, -21.3886089, -3.0931373, -14.4566574, 14.5838356
14: -22.9158096, -5.0338163, -22.9088707, -5.0425301, -16.8524399, 16.8825989
15: -9.0461750, 3.5653877, -9.0559187, 3.4896007, -9.8578148, 9.9414215
16: -9.6312561, 1.2228718, -9.6505966, 1.1619325, -10.0925636, 10.1969986
17: -20.8556957, -4.1644611, -20.8318214, -4.1644635, -13.7468872, 13.7351074
18: -3.2701285, 11.7987690, -3.2220993, 11.8024979, -11.1749954, 11.1203499
19: 1.7484767, 11.0726871, 1.8387380, 11.0891714, -9.2890015, 9.1741829
20: -0.8699195, 9.8770046, -0.8106742, 9.8869896, -10.7569094, 10.6876793
21: 0.6482952, 13.1035566, 0.7457449, 13.1187077, -12.3595047, 12.2273483
22: 1.9261875, 12.2776461, 1.9746723, 12.2922783, -8.4457321, 8.3677254
23: 0.0475507, 11.0555086, 0.1877646, 11.0897598, -9.6440735, 9.4673882
24: -5.5075717, 9.4823895, -5.3804708, 9.5127363, -12.1762466, 12.0204849
25: -4.5120401, 9.7138329, -4.4068379, 9.7448330, -11.9802780, 11.8445778
26: 2.8774538, 16.2842407, 2.9692750, 16.3014336, -13.4239798, 13.3149662
27: -0.0044215, 12.2053137, 0.0836682, 12.2218647, -10.5066109, 10.4074135
28: 0.6167667, 12.6254501, 0.7196057, 12.6509476, -11.6208954, 11.4907913
29: -0.5343219, 8.9894133, -0.4790502, 9.0038967, -6.7836647, 6.7139397
30: -4.1243877, 9.9486561, -4.0289755, 9.9744377, -12.7725525, 12.6571503
31: -3.1796720, 11.8028631, -3.0798016, 11.8184013, -11.5133171, 11.3882027
32: -19.0222111, -5.8230801, -19.0405197, -5.8392787, -9.3784065, 9.4334641
33: -38.4898071, -16.8451614, -38.4322815, -16.8390503, -15.9974518, 15.8983917
34: -37.8430786, -23.4299011, -37.8564186, -23.4214821, -10.3240967, 10.3187408
35: -29.0621300, -14.0619850, -29.0211334, -14.0570459, -11.8494339, 11.7786255
36: -22.0740795, -9.1958380, -22.0467529, -9.1992445, -9.2562828, 9.2220592
37: -39.7379494, -18.9985924, -39.6492538, -18.9861755, -15.8103180, 15.7041626
38: -36.0551338, -19.3198814, -36.0327415, -19.3390942, -14.5310135, 14.4685364
39: -38.4269257, -16.9099178, -38.3921509, -16.9169922, -14.8399887, 14.8096237
40: -34.3416214, -20.4502411, -34.3721848, -20.4453773, -8.6580315, 8.6850719
41: -21.2159290, -5.2952771, -21.2063484, -5.2955136, -12.3767471, 12.3803406
42: -23.4797268, -11.3948936, -23.4698315, -11.3957701, -9.8710632, 9.8576736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=64, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A1_A2_A2_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3413645, upper bound: 6.2910660
time: 12.42 seconds

## Relational analysis of IS_A1_A2_A2_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3268478
time: 12.51 seconds

## BFS IS instance: IS_A2_A2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -26.9801788, -9.4066477, -26.9782295, -9.4919662, -12.0035667, 12.0501938
1: -9.7916336, 0.0475738, -9.7925224, -0.0112257, -6.8236065, 6.8622227
2: -4.3740282, 4.9136391, -4.3695803, 4.8837733, -6.8788452, 6.9284592
3: -13.4826164, -0.5191851, -13.4811192, -0.6294911, -9.6021156, 9.7132721
4: -5.4894714, 7.4400635, -5.4862928, 7.3371477, -8.9554482, 9.0412140
5: -8.9342155, 4.2975016, -8.9302006, 4.1931844, -11.4842682, 11.5903397
6: -24.1244087, -8.9030523, -24.1184978, -8.9189453, -9.8232422, 9.8583183
7: -9.6921139, 2.7414479, -9.6886797, 2.6889014, -8.8993530, 8.9445229
8: -12.3394108, 3.1461418, -12.3343916, 3.1001697, -9.4541740, 9.4860687
9: -7.0685959, 8.7582998, -7.0722017, 8.6211081, -10.8754005, 10.9994316
10: -7.0419760, 7.3153281, -7.0402985, 7.1749496, -11.5520248, 11.6914177
11: -4.7760768, 5.0614028, -4.6715798, 5.0572271, -8.2893105, 8.1837044
12: -16.8906994, -0.5621355, -16.8734264, -0.5926595, -11.4189339, 11.4730339
13: -21.3865986, -2.9689636, -21.3832245, -3.1013899, -14.4738007, 14.6119194
14: -22.9411621, -5.0204382, -22.9009209, -5.0484743, -16.8987503, 16.8875809
15: -9.0407610, 3.5694096, -9.0377674, 3.4728842, -9.8334579, 9.9255371
16: -9.6382504, 1.2077599, -9.6408892, 1.1385586, -10.0935173, 10.1734276
17: -20.8593788, -4.1565671, -20.8116245, -4.1784186, -13.7386475, 13.7224197
18: -3.2941029, 11.8066177, -3.2231061, 11.8047829, -11.2009506, 11.1241264
19: 1.7427325, 11.0800896, 1.8485768, 11.0840874, -9.2887497, 9.1851273
20: -0.8866174, 9.8848877, -0.8027401, 9.8847618, -10.7713795, 10.6876278
21: 0.6306543, 13.1190338, 0.7536323, 13.1160603, -12.3630142, 12.2244110
22: 1.9182539, 12.2904091, 1.9861412, 12.2907572, -8.4521790, 8.3816795
23: 0.0284758, 11.0793619, 0.2069557, 11.0825491, -9.6589050, 9.4781342
24: -5.5299125, 9.5060987, -5.3651114, 9.5082989, -12.1965103, 12.0247574
25: -4.5278211, 9.7385893, -4.3879700, 9.7375431, -11.9918518, 11.8554001
26: 2.8520489, 16.3059464, 2.9839230, 16.3011971, -13.4491482, 13.3220234
27: -0.0277715, 12.2169552, 0.0945582, 12.2171497, -10.5280228, 10.4076767
28: 0.5985126, 12.6450701, 0.7357931, 12.6433401, -11.6339569, 11.5096207
29: -0.5440092, 9.0082893, -0.4682285, 9.0028744, -6.7932167, 6.7300472
30: -4.1528530, 9.9735508, -4.0151596, 9.9706154, -12.8098907, 12.6856995
31: -3.1935060, 11.8121214, -3.0714242, 11.8138237, -11.5142670, 11.3998451
32: -19.0491829, -5.8100872, -19.0452232, -5.8418570, -9.4000511, 9.4582901
33: -38.4839554, -16.8475189, -38.4249573, -16.8480721, -15.9826355, 15.8894882
34: -37.8706551, -23.4172821, -37.8577957, -23.4243145, -10.3350906, 10.3124924
35: -29.0508442, -14.0674067, -29.0136070, -14.0691910, -11.8361168, 11.7704010
36: -22.0518894, -9.2100506, -22.0301895, -9.2160892, -9.2498131, 9.2081394
37: -39.7369423, -18.9963131, -39.6338120, -18.9967766, -15.7882538, 15.6885452
38: -36.0456161, -19.3290768, -36.0231934, -19.3527775, -14.4939346, 14.4607773
39: -38.4087105, -16.9236984, -38.3819809, -16.9283829, -14.7808304, 14.7974167
40: -34.3773613, -20.4321365, -34.3741150, -20.4576454, -8.6736984, 8.7056808
41: -21.2330971, -5.2865958, -21.2087212, -5.2985945, -12.3891983, 12.4032784
42: -23.4876118, -11.3893814, -23.4706669, -11.3951502, -9.8737831, 9.8584938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=64, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A1_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3090601
time: 12.02 seconds

## Relational analysis of IS_A2_A2_A1_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3448448
time: 15.12 seconds

## BFS IS instance: IS_A2_A2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -26.9843025, -9.4061623, -26.9931221, -9.4710302, -12.0277252, 12.0625496
1: -9.7928524, 0.0479126, -9.7976303, -0.0012546, -6.8376560, 6.8705139
2: -4.3750048, 4.9138355, -4.3736072, 4.8893681, -6.8854752, 6.9329147
3: -13.4864292, -0.5190825, -13.4943752, -0.6116047, -9.6244202, 9.7242432
4: -5.4943366, 7.4401560, -5.5038099, 7.3544555, -8.9777374, 9.0565109
5: -8.9382572, 4.2975936, -8.9444447, 4.2085700, -11.5039215, 11.6024094
6: -24.1241817, -8.9027939, -24.1189365, -8.9184351, -9.8265724, 9.8594570
7: -9.6944704, 2.7415605, -9.6973648, 2.7001414, -8.9132004, 8.9513702
8: -12.3423300, 3.1462030, -12.3464899, 3.1103201, -9.4669914, 9.4981041
9: -7.0747776, 8.7584362, -7.0938983, 8.6460218, -10.9065475, 11.0176430
10: -7.0501156, 7.3155622, -7.0686975, 7.2070503, -11.5921211, 11.7140923
11: -4.7763066, 5.0636344, -4.6803799, 5.0665526, -8.2990799, 8.1965904
12: -16.8907967, -0.5600818, -16.8844986, -0.5842603, -11.4266472, 11.4844894
13: -21.3867836, -2.9664502, -21.3927860, -3.0907893, -14.4849319, 14.6129303
14: -22.9426346, -5.0168839, -22.9108238, -5.0359440, -16.9099045, 16.8966522
15: -9.0462227, 3.5696149, -9.0591488, 3.4914422, -9.8576088, 9.9451218
16: -9.6426287, 1.2081590, -9.6558266, 1.1625905, -10.1219635, 10.1859970
17: -20.8601418, -4.1506896, -20.8333435, -4.1575923, -13.7556229, 13.7497864
18: -3.2944353, 11.8060780, -3.2239556, 11.8052473, -11.2023544, 11.1245728
19: 1.7423522, 11.0828209, 1.8373907, 11.0937290, -9.2982674, 9.1989708
20: -0.8874197, 9.8866673, -0.8128064, 9.8908787, -10.7782984, 10.6994734
21: 0.6298580, 13.1212845, 0.7439969, 13.1253452, -12.3742371, 12.2420807
22: 1.9177732, 12.2923651, 1.9739408, 12.2975178, -8.4582748, 8.3958435
23: 0.0282235, 11.0847921, 0.1864940, 11.1016340, -9.6748199, 9.5039024
24: -5.5301352, 9.5102463, -5.3813615, 9.5233431, -12.2097015, 12.0450554
25: -4.5282717, 9.7438145, -4.4078856, 9.7562370, -12.0083923, 11.8805618
26: 2.8510642, 16.3082123, 2.9674315, 16.3095055, -13.4584408, 13.3407803
27: -0.0281465, 12.2199163, 0.0825911, 12.2277193, -10.5354156, 10.4160500
28: 0.5979881, 12.6497650, 0.7179661, 12.6605940, -11.6501083, 11.5317726
29: -0.5443516, 9.0102997, -0.4793469, 9.0101881, -6.7999821, 6.7434902
30: -4.1534381, 9.9773102, -4.0302505, 9.9846268, -12.8226013, 12.7021179
31: -3.1939926, 11.8143101, -3.0814331, 11.8235435, -11.5241089, 11.4145432
32: -19.0493069, -5.8095541, -19.0481586, -5.8386164, -9.4042969, 9.4616928
33: -38.4842186, -16.8448658, -38.4338875, -16.8381844, -15.9923782, 15.9019051
34: -37.8713875, -23.4161949, -37.8612633, -23.4185257, -10.3449707, 10.3163223
35: -29.0509396, -14.0639658, -29.0224323, -14.0569029, -11.8476486, 11.7810402
36: -22.0520096, -9.2053223, -22.0473557, -9.1997890, -9.2614861, 9.2209930
37: -39.7370987, -18.9924812, -39.6506424, -18.9831181, -15.8006516, 15.7093582
38: -36.0457802, -19.3265305, -36.0338821, -19.3399734, -14.5036163, 14.4718399
39: -38.4089813, -16.9209328, -38.3926468, -16.9185715, -14.7888794, 14.8068695
40: -34.3803329, -20.4319153, -34.3851776, -20.4455357, -8.6900864, 8.7142277
41: -21.2334213, -5.2856526, -21.2111015, -5.2941685, -12.3945465, 12.4074554
42: -23.4868279, -11.3889055, -23.4700470, -11.3934698, -9.8749847, 9.8636360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=64, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A1_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3090601
time: 23.87 seconds

## Relational analysis of IS_A2_A2_A1_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3448448
time: 25.45 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -26.9549370, -9.4212179, -26.9857807, -9.4902248, -11.9339142, 12.0468407
1: -9.7736635, 0.0341902, -9.7936106, -0.0101910, -6.7770672, 6.8571930
2: -4.3613572, 4.9040937, -4.3703828, 4.8844523, -6.8602333, 6.9122391
3: -13.4575386, -0.5502243, -13.4884663, -0.6291504, -9.5767822, 9.6901054
4: -5.4605546, 7.4194183, -5.4967718, 7.3373985, -8.9111671, 9.0292740
5: -8.9093409, 4.2687812, -8.9388638, 4.1932940, -11.4582748, 11.5702438
6: -24.1019268, -8.9121628, -24.1175690, -8.9189072, -9.8120613, 9.8452568
7: -9.6741514, 2.7362361, -9.6925821, 2.6893897, -8.8747101, 8.9439964
8: -12.3174782, 3.1351135, -12.3393841, 3.1004500, -9.4235439, 9.4710236
9: -7.0417118, 8.7335453, -7.0853233, 8.6215162, -10.8297653, 10.9887199
10: -7.0213442, 7.3067532, -7.0617237, 7.1753759, -11.5313644, 11.7127304
11: -4.7287340, 5.0344229, -4.6721020, 5.0608644, -8.2506256, 8.1584892
12: -16.8844376, -0.6003031, -16.8738117, -0.5876496, -11.4281006, 11.4338722
13: -21.3600883, -3.0664744, -21.3791046, -3.0945692, -14.4258957, 14.5219536
14: -22.9148235, -5.0496168, -22.9045067, -5.0395012, -16.8518753, 16.8694534
15: -9.0244379, 3.5475273, -9.0523863, 3.4733005, -9.8182831, 9.9214745
16: -9.6230059, 1.2287455, -9.6507607, 1.1399989, -10.0651360, 10.1921539
17: -20.8577271, -4.1895318, -20.8136635, -4.1643209, -13.7554169, 13.6942596
18: -3.2458782, 11.7941055, -3.2242920, 11.8027840, -11.1487198, 11.1206398
19: 1.7767208, 11.0617447, 1.8477292, 11.0900536, -9.2638206, 9.1526451
20: -0.8491988, 9.8670902, -0.8046961, 9.8873930, -10.7365913, 10.6717863
21: 0.6894059, 13.0888138, 0.7514117, 13.1192236, -12.3217392, 12.2004013
22: 1.9319854, 12.2749338, 1.9845848, 12.2936583, -8.4379349, 8.3561172
23: 0.0788157, 11.0464535, 0.2065710, 11.0946751, -9.6212463, 9.4342918
24: -5.4690824, 9.4694538, -5.3656831, 9.5151176, -12.1422729, 11.9887848
25: -4.5016623, 9.7096653, -4.3890934, 9.7489471, -11.9772186, 11.8200569
26: 2.9064808, 16.2706890, 2.9812198, 16.3030472, -13.3965664, 13.2894688
27: 0.0318239, 12.1894760, 0.0933259, 12.2223129, -10.4686890, 10.3861084
28: 0.6417739, 12.6159344, 0.7347095, 12.6534758, -11.6005936, 11.4641724
29: -0.5221746, 8.9861746, -0.4693409, 9.0053921, -6.7743969, 6.6991901
30: -4.0870423, 9.9346228, -4.0166597, 9.9757690, -12.7394180, 12.6339188
31: -3.1453636, 11.7924004, -3.0726042, 11.8169832, -11.4801788, 11.3669128
32: -19.0307255, -5.8317041, -19.0461693, -5.8403001, -9.3816872, 9.4296303
33: -38.4727020, -16.8575497, -38.4256897, -16.8409920, -15.9687653, 15.8818893
34: -37.8523636, -23.4272861, -37.8611107, -23.4233532, -10.3157349, 10.3318558
35: -29.0540314, -14.0735016, -29.0138206, -14.0596428, -11.8233032, 11.7664490
36: -22.0729103, -9.2188044, -22.0320988, -9.2021809, -9.2500610, 9.2078762
37: -39.7149200, -19.0169983, -39.6342201, -18.9874840, -15.7843475, 15.6725616
38: -36.0549240, -19.3446007, -36.0266380, -19.3416214, -14.5134811, 14.4763222
39: -38.4160614, -16.9277306, -38.3830833, -16.9187241, -14.8285675, 14.7970734
40: -34.3527794, -20.4351139, -34.3779182, -20.4564095, -8.6508026, 8.7107105
41: -21.2105923, -5.2982140, -21.2093430, -5.2947540, -12.3746338, 12.3820572
42: -23.4724445, -11.4051447, -23.4710312, -11.3948221, -9.8772316, 9.8424263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A2_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2692902
time: 21.48 seconds

## Relational analysis of IS_A2_A2_A2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A2_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3050736
time: 19.57 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -26.9705505, -9.3991938, -26.9904137, -9.4897566, -11.9469490, 12.0733566
1: -9.7796412, 0.0452874, -9.7949486, -0.0098486, -6.7853889, 6.8714294
2: -4.3655305, 4.9100976, -4.3714991, 4.8846402, -6.8648796, 6.9193363
3: -13.4708796, -0.5322509, -13.4922743, -0.6290228, -9.5879478, 9.7125130
4: -5.4781508, 7.4368052, -5.5017033, 7.3375416, -8.9265327, 9.0516968
5: -8.9237309, 4.2842326, -8.9429188, 4.1933861, -11.4704895, 11.5899582
6: -24.1027088, -8.9114218, -24.1174812, -8.9186382, -9.8133812, 9.8487206
7: -9.6829166, 2.7477207, -9.6949806, 2.6895123, -8.8819771, 8.9579353
8: -12.3302021, 3.1463575, -12.3427200, 3.1005173, -9.4358292, 9.4850082
9: -7.0634766, 8.7584944, -7.0915709, 8.6216049, -10.8480453, 11.0199585
10: -7.0498409, 7.3389692, -7.0699229, 7.1756053, -11.5541191, 11.7529602
11: -4.7381096, 5.0441704, -4.6723385, 5.0633740, -8.2636147, 8.1682739
12: -16.8955021, -0.5921227, -16.8739147, -0.5853769, -11.4406891, 11.4417877
13: -21.3699608, -3.0556040, -21.3792820, -3.0916901, -14.4300194, 14.5333672
14: -22.9249077, -5.0370388, -22.9060173, -5.0359373, -16.8614197, 16.8808823
15: -9.0458269, 3.5661466, -9.0578814, 3.4735048, -9.8379288, 9.9457092
16: -9.6381254, 1.2529211, -9.6551952, 1.1403913, -10.0779343, 10.2207527
17: -20.8795338, -4.1685739, -20.8144531, -4.1584277, -13.7832794, 13.7113342
18: -3.2468207, 11.7946606, -3.2246332, 11.8022051, -11.1508179, 11.1221733
19: 1.7653975, 11.0714111, 1.8473473, 11.0928097, -9.2777901, 9.1621971
20: -0.8597238, 9.8732452, -0.8054967, 9.8892021, -10.7489262, 10.6787415
21: 0.6782808, 13.0985069, 0.7505841, 13.1216116, -12.3395386, 12.2116585
22: 1.9196672, 12.2817335, 1.9840951, 12.2955761, -8.4522095, 8.3622894
23: 0.0582774, 11.0655718, 0.2062727, 11.1001291, -9.6472015, 9.4506874
24: -5.4853506, 9.4845400, -5.3659091, 9.5193071, -12.1626740, 12.0021591
25: -4.5216370, 9.7284555, -4.3895416, 9.7542572, -12.0025063, 11.8366508
26: 2.8888812, 16.2797794, 2.9802489, 16.3054371, -13.4165554, 13.2995300
27: 0.0193584, 12.2003727, 0.0929837, 12.2254505, -10.4799843, 10.3932838
28: 0.6238458, 12.6332245, 0.7341580, 12.6582136, -11.6229248, 11.4804420
29: -0.5333883, 8.9935799, -0.4696821, 9.0074158, -6.7879333, 6.7060280
30: -4.1021743, 9.9486790, -4.0172052, 9.9795656, -12.7567139, 12.6466904
31: -3.1557558, 11.8021727, -3.0730743, 11.8192291, -11.4951668, 11.3768463
32: -19.0339241, -5.8284483, -19.0464973, -5.8397322, -9.3851547, 9.4339638
33: -38.4818649, -16.8475494, -38.4259911, -16.8383198, -15.9813385, 15.8917007
34: -37.8559113, -23.4214058, -37.8618851, -23.4222260, -10.3196182, 10.3419724
35: -29.0632591, -14.0610352, -29.0139122, -14.0562868, -11.8346710, 11.7781181
36: -22.0901833, -9.2024555, -22.0322037, -9.1974287, -9.2667236, 9.2195263
37: -39.7317963, -19.0033169, -39.6343880, -18.9836235, -15.8052368, 15.6851273
38: -36.0649414, -19.3346977, -36.0268478, -19.3379555, -14.5261841, 14.4855080
39: -38.4267998, -16.9178047, -38.3833160, -16.9159012, -14.8392410, 14.8045959
40: -34.3639717, -20.4228096, -34.3808708, -20.4562073, -8.6597214, 8.7282009
41: -21.2130585, -5.2936959, -21.2097454, -5.2937565, -12.3787003, 12.3876038
42: -23.4720039, -11.4033890, -23.4702568, -11.3943405, -9.8824234, 9.8437653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A2_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2720883
time: 20.88 seconds

## Relational analysis of IS_A2_A2_A2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A2_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3078721
time: 27.72 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -26.9673538, -9.4070511, -26.9889812, -9.4898348, -11.9957314, 12.0639725
1: -9.7838173, 0.0466011, -9.7963943, -0.0098472, -6.8211441, 6.8721046
2: -4.3698382, 4.9153047, -4.3725948, 4.8846436, -6.8793144, 6.9257526
3: -13.4708576, -0.5319710, -13.4920845, -0.6289196, -9.5900345, 9.7120514
4: -5.4786844, 7.4391642, -5.5015383, 7.3375425, -8.9434700, 9.0539742
5: -8.9220505, 4.2856483, -8.9422188, 4.1934872, -11.4720078, 11.5907478
6: -24.1113510, -8.9066401, -24.1180725, -8.9177160, -9.8136597, 9.8569508
7: -9.6871414, 2.7500057, -9.6960363, 2.6894932, -8.8925972, 8.9612579
8: -12.3321123, 3.1497042, -12.3432970, 3.1006074, -9.4450226, 9.4887600
9: -7.0626450, 8.7568121, -7.0910378, 8.6217251, -10.8676300, 11.0169678
10: -7.0346279, 7.3201284, -7.0646901, 7.1757870, -11.5432014, 11.7193031
11: -4.7421508, 5.0468478, -4.6724195, 5.0639787, -8.2634888, 8.1698227
12: -16.8987637, -0.5919132, -16.8740997, -0.5855691, -11.4349136, 11.4417877
13: -21.3750725, -3.0439491, -21.3831291, -3.0939817, -14.4486351, 14.5478058
14: -22.9292603, -5.0448942, -22.9052410, -5.0384493, -16.8845520, 16.8750458
15: -9.0325937, 3.5601997, -9.0545187, 3.4737966, -9.8261070, 9.9312706
16: -9.6370964, 1.2405467, -9.6546001, 1.1402357, -10.0951653, 10.2076378
17: -20.8764725, -4.1738229, -20.8143082, -4.1601663, -13.7717743, 13.7076035
18: -3.2647793, 11.8035088, -3.2248654, 11.8052197, -11.1700897, 11.1244202
19: 1.7692733, 11.0694513, 1.8474183, 11.0921803, -9.2737236, 9.1748924
20: -0.8599651, 9.8708782, -0.8053999, 9.8883553, -10.7483206, 10.6762781
21: 0.6810315, 13.0945044, 0.7508786, 13.1206665, -12.3391113, 12.2108498
22: 1.9197445, 12.2850113, 1.9842639, 12.2964125, -8.4528732, 8.3805676
23: 0.0599766, 11.0614929, 0.2062461, 11.0987091, -9.6440926, 9.4573059
24: -5.4899096, 9.4885483, -5.3660378, 9.5203171, -12.1682281, 12.0072403
25: -4.5223880, 9.7282524, -4.3894796, 9.7538509, -12.0029335, 11.8442993
26: 2.8880014, 16.2806606, 2.9804807, 16.3056183, -13.4176168, 13.3001804
27: 0.0171328, 12.1984825, 0.0930669, 12.2247448, -10.4824677, 10.3913918
28: 0.6264086, 12.6295509, 0.7341940, 12.6570892, -11.6193695, 11.4940643
29: -0.5356356, 9.0007620, -0.4694529, 9.0089474, -6.7914314, 6.7229042
30: -4.1078916, 9.9520998, -4.0170851, 9.9805717, -12.7649231, 12.6599579
31: -3.1613908, 11.8050833, -3.0728383, 11.8204107, -11.4994202, 11.3923759
32: -19.0362682, -5.8304634, -19.0465717, -5.8399472, -9.3892975, 9.4390030
33: -38.4839630, -16.8484726, -38.4262848, -16.8390961, -15.9800339, 15.8919525
34: -37.8669586, -23.4171333, -37.8614883, -23.4206009, -10.3254700, 10.3223419
35: -29.0613289, -14.0647106, -29.0142479, -14.0579205, -11.8320160, 11.7747803
36: -22.0750179, -9.2155266, -22.0322056, -9.2013474, -9.2738152, 9.2140121
37: -39.7305031, -19.0051746, -39.6346741, -18.9845924, -15.7928619, 15.6818314
38: -36.0585136, -19.3441868, -36.0264473, -19.3411102, -14.5027466, 14.4768486
39: -38.4190369, -16.9279575, -38.3828316, -16.9186611, -14.7987823, 14.7961349
40: -34.3567047, -20.4356155, -34.3784332, -20.4567146, -8.6581650, 8.7136650
41: -21.2164116, -5.2924910, -21.2099438, -5.2937546, -12.3808899, 12.3966675
42: -23.4810333, -11.3972549, -23.4714317, -11.3931055, -9.8729362, 9.8495674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A2_A1_A2_A1_B1

### Relational analysis result of IS_A2_A2_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2789102
time: 20.86 seconds

## Relational analysis of IS_A2_A2_A2_A1_A2_A1_B2

### Relational analysis result of IS_A2_A2_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3146970
time: 22.28 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -26.9829865, -9.3849754, -26.9936390, -9.4893188, -12.0087624, 12.0904961
1: -9.7898083, 0.0577195, -9.7977314, -0.0094953, -6.8294773, 6.8863297
2: -4.3740253, 4.9212861, -4.3737268, 4.8848238, -6.8839531, 6.9328651
3: -13.4842081, -0.5140052, -13.4959068, -0.6287966, -9.6012039, 9.7344627
4: -5.4962945, 7.4565392, -5.5064712, 7.3376989, -8.9588509, 9.0763702
5: -8.9364338, 4.3010931, -8.9462662, 4.1935606, -11.4841919, 11.6104774
6: -24.1121769, -8.9058981, -24.1180153, -8.9174461, -9.8149796, 9.8604012
7: -9.6959019, 2.7614975, -9.6984510, 2.6896348, -8.8998680, 8.9752083
8: -12.3448591, 3.1609824, -12.3466425, 3.1006720, -9.4572906, 9.5027828
9: -7.0843444, 8.7817612, -7.0972762, 8.6218204, -10.8859062, 11.0481949
10: -7.0631504, 7.3523149, -7.0728617, 7.1760240, -11.5659409, 11.7595367
11: -4.7515316, 5.0565844, -4.6726379, 5.0664716, -8.2764473, 8.1796074
12: -16.9098358, -0.5837196, -16.8742104, -0.5832769, -11.4475861, 11.4496956
13: -21.3848972, -3.0330238, -21.3833351, -3.0912113, -14.4527512, 14.5592041
14: -22.9393845, -5.0323315, -22.9067039, -5.0348549, -16.8940964, 16.8864746
15: -9.0539818, 3.5788035, -9.0600185, 3.4740138, -9.8457527, 9.9555283
16: -9.6521988, 1.2647157, -9.6590176, 1.1406603, -10.1079178, 10.2362328
17: -20.8982906, -4.1528745, -20.8150749, -4.1542211, -13.7996063, 13.7246857
18: -3.2657206, 11.8039894, -3.2251861, 11.8046474, -11.1721725, 11.1259651
19: 1.7579761, 11.0791206, 1.8470292, 11.0949354, -9.2876701, 9.1844139
20: -0.8704615, 9.8770180, -0.8062150, 9.8901939, -10.7606554, 10.6832333
21: 0.6699104, 13.1041622, 0.7500498, 13.1230545, -12.3568954, 12.2221146
22: 1.9073949, 12.2917976, 1.9838018, 12.2983618, -8.4671478, 8.3867493
23: 0.0394182, 11.0806103, 0.2059681, 11.1041851, -9.6700439, 9.4736938
24: -5.5062141, 9.5036192, -5.3662844, 9.5245199, -12.1886215, 12.0205498
25: -4.5423679, 9.7470312, -4.3898916, 9.7591534, -12.0282211, 11.8609123
26: 2.8704047, 16.2897148, 2.9795232, 16.3080215, -13.4376163, 13.3101921
27: 0.0046804, 12.2094212, 0.0927153, 12.2278748, -10.4937973, 10.3985786
28: 0.6084845, 12.6468143, 0.7336462, 12.6618271, -11.6416931, 11.5103264
29: -0.5468612, 9.0081367, -0.4697775, 9.0109825, -6.8049469, 6.7297363
30: -4.1230268, 9.9661522, -4.0176859, 9.9843683, -12.7822037, 12.6727142
31: -3.1717815, 11.8148441, -3.0733440, 11.8226709, -11.5144272, 11.4022865
32: -19.0394707, -5.8271961, -19.0468674, -5.8393941, -9.3927536, 9.4433022
33: -38.4931374, -16.8384819, -38.4265709, -16.8363972, -15.9925919, 15.9018059
34: -37.8704872, -23.4112644, -37.8622513, -23.4194603, -10.3293610, 10.3324699
35: -29.0705509, -14.0522614, -29.0142994, -14.0545473, -11.8433990, 11.7864914
36: -22.0922718, -9.1991863, -22.0323067, -9.1965961, -9.2904243, 9.2256851
37: -39.7473869, -18.9914932, -39.6348305, -18.9807014, -15.8137360, 15.6943817
38: -36.0685310, -19.3342762, -36.0266151, -19.3375072, -14.5154266, 14.4860229
39: -38.4298286, -16.9180679, -38.3831596, -16.9158077, -14.8094406, 14.8036499
40: -34.3678665, -20.4233208, -34.3814125, -20.4565010, -8.6670952, 8.7311516
41: -21.2189064, -5.2880111, -21.2103806, -5.2928042, -12.3849792, 12.4022484
42: -23.4806137, -11.3955097, -23.4706860, -11.3926344, -9.8781242, 9.8509102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A2_A1_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2817079
time: 25.83 seconds

## Relational analysis of IS_A2_A2_A2_A1_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3174948
time: 20.65 seconds

## BFS IS instance: IS_A2_A2_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -26.9638176, -9.3963337, -26.9858570, -9.4905844, -11.9405670, 12.0759964
1: -9.7798758, 0.0512521, -9.7936697, -0.0104179, -6.7828674, 6.8755550
2: -4.3647432, 4.9093857, -4.3702731, 4.8842721, -6.8600464, 6.9265366
3: -13.4700251, -0.5164490, -13.4887753, -0.6293068, -9.5889435, 9.7242126
4: -5.4763026, 7.4464769, -5.4968810, 7.3373752, -8.9268570, 9.0563316
5: -8.9236832, 4.3003974, -8.9390860, 4.1932716, -11.4716415, 11.6020050
6: -24.1051826, -8.9093742, -24.1167431, -8.9190636, -9.8164597, 9.8480244
7: -9.6810474, 2.7466698, -9.6925077, 2.6892157, -8.8818703, 8.9544868
8: -12.3248940, 3.1438324, -12.3392754, 3.1003020, -9.4297295, 9.4809246
9: -7.0553913, 8.7718287, -7.0856466, 8.6213779, -10.8437767, 11.0272293
10: -7.0365267, 7.3454552, -7.0621819, 7.1752958, -11.5462761, 11.7527771
11: -4.7720380, 5.0519862, -4.6720409, 5.0612221, -8.2939377, 8.1757050
12: -16.8934898, -0.5686349, -16.8734455, -0.5877100, -11.4352417, 11.4667473
13: -21.3895111, -2.9894934, -21.3798752, -3.0946732, -14.4545593, 14.5992470
14: -22.9319801, -5.0258875, -22.9043961, -5.0396852, -16.8691254, 16.8931351
15: -9.0420094, 3.5808911, -9.0522966, 3.4731812, -9.8362541, 9.9563713
16: -9.6277637, 1.2362547, -9.6507549, 1.1396270, -10.0652924, 10.2128487
17: -20.8711624, -4.1661282, -20.8134193, -4.1644692, -13.7698746, 13.7154846
18: -3.2784722, 11.8047638, -3.2240963, 11.8030005, -11.1816025, 11.1310921
19: 1.7346621, 11.0761929, 1.8478355, 11.0905638, -9.3061523, 9.1668129
20: -0.8831186, 9.8779068, -0.8044758, 9.8877907, -10.7709093, 10.6823826
21: 0.6346304, 13.1067791, 0.7516618, 13.1197662, -12.3769302, 12.2178650
22: 1.9130039, 12.2809086, 1.9849172, 12.2937469, -8.4627914, 8.3596039
23: 0.0180070, 11.0696459, 0.2065691, 11.0953636, -9.6825256, 9.4574165
24: -5.5299969, 9.4910641, -5.3656292, 9.5158319, -12.2036362, 12.0104294
25: -4.5379763, 9.7256374, -4.3889642, 9.7492571, -12.0137672, 11.8361893
26: 2.8539939, 16.2886181, 2.9814448, 16.3035851, -13.4495907, 13.3071728
27: -0.0200419, 12.2069235, 0.0935512, 12.2229500, -10.5213051, 10.4034309
28: 0.5926983, 12.6356564, 0.7347720, 12.6540232, -11.6499329, 11.4838371
29: -0.5459807, 8.9952898, -0.4691420, 9.0055676, -6.7984524, 6.7084217
30: -4.1430988, 9.9561396, -4.0165157, 9.9764509, -12.7956085, 12.6557465
31: -3.1910992, 11.8064375, -3.0723438, 11.8173141, -11.5258636, 11.3807983
32: -19.0395508, -5.8101549, -19.0462646, -5.8403249, -9.3898239, 9.4527359
33: -38.4948387, -16.8476105, -38.4255981, -16.8409271, -16.0035553, 15.8884811
34: -37.8533859, -23.4220886, -37.8601036, -23.4234161, -10.3248100, 10.3333817
35: -29.0646019, -14.0677948, -29.0137482, -14.0596399, -11.8507957, 11.7675743
36: -22.0795631, -9.2065058, -22.0306396, -9.2021866, -9.2586174, 9.2093124
37: -39.7518044, -19.0015945, -39.6341171, -18.9872818, -15.8226929, 15.6847000
38: -36.0641098, -19.3211536, -36.0242844, -19.3416386, -14.5365448, 14.4695892
39: -38.4303589, -16.9159641, -38.3832169, -16.9187927, -14.8436966, 14.8065872
40: -34.3557701, -20.4328384, -34.3776970, -20.4566994, -8.6534233, 8.7125492
41: -21.2258720, -5.2881222, -21.2091999, -5.2953835, -12.3851318, 12.3933144
42: -23.4805756, -11.3944817, -23.4708176, -11.3949633, -9.8852043, 9.8526649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A2_A2_A1_A1_B1

### Relational analysis result of IS_A2_A2_A2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3441629, upper bound: 6.2959553
time: 27.85 seconds

## Relational analysis of IS_A2_A2_A2_A2_A1_A1_B2

### Relational analysis result of IS_A2_A2_A2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3317381
time: 26.68 seconds

## BFS IS instance: IS_A2_A2_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -26.9794579, -9.3742752, -26.9905205, -9.4900684, -11.9535980, 12.1024895
1: -9.7858515, 0.0623584, -9.7950106, -0.0100667, -6.7911949, 6.8897858
2: -4.3689060, 4.9153547, -4.3714099, 4.8844814, -6.8646927, 6.9336395
3: -13.4833488, -0.4984827, -13.4925842, -0.6291742, -9.6001091, 9.7466507
4: -5.4938922, 7.4638577, -5.5018315, 7.3374844, -8.9422417, 9.0787125
5: -8.9380493, 4.3158646, -8.9431419, 4.1933641, -11.4838638, 11.6217346
6: -24.1060085, -8.9086218, -24.1166687, -8.9187765, -9.8177910, 9.8514919
7: -9.6898165, 2.7581639, -9.6949110, 2.6893306, -8.8891144, 8.9684181
8: -12.3376503, 3.1551163, -12.3426218, 3.1003916, -9.4420071, 9.4949245
9: -7.0771556, 8.7967920, -7.0918789, 8.6214600, -10.8620377, 11.0584488
10: -7.0649943, 7.3776593, -7.0703402, 7.1755342, -11.5690079, 11.7930107
11: -4.7814345, 5.0617399, -4.6722903, 5.0637321, -8.3068924, 8.1854744
12: -16.9045506, -0.5604378, -16.8735390, -0.5854197, -11.4479103, 11.4746780
13: -21.3993835, -2.9785643, -21.3800926, -3.0918941, -14.4587173, 14.6106567
14: -22.9421234, -5.0133438, -22.9058895, -5.0361242, -16.8786774, 16.9045792
15: -9.0634165, 3.5995035, -9.0578318, 3.4733713, -9.8559074, 9.9806137
16: -9.6428461, 1.2603989, -9.6551933, 1.1400237, -10.0780373, 10.2414169
17: -20.8930206, -4.1452036, -20.8141804, -4.1585727, -13.7977524, 13.7326050
18: -3.2794263, 11.8053169, -3.2243955, 11.8024368, -11.1836929, 11.1326447
19: 1.7233629, 11.0858545, 1.8474641, 11.0933075, -9.3200951, 9.1763458
20: -0.8936431, 9.8840599, -0.8052876, 9.8896065, -10.7832499, 10.6893473
21: 0.6235285, 13.1164961, 0.7508490, 13.1221600, -12.3947372, 12.2291412
22: 1.9006948, 12.2876892, 1.9844418, 12.2957029, -8.4770660, 8.3657913
23: -0.0025406, 11.0887909, 0.2063006, 11.1008396, -9.7084656, 9.4738159
24: -5.5463033, 9.5061359, -5.3658285, 9.5200424, -12.2240448, 12.0237427
25: -4.5579543, 9.7443962, -4.3893995, 9.7545414, -12.0390396, 11.8528214
26: 2.8364305, 16.2976856, 2.9804544, 16.3059692, -13.4695387, 13.3172312
27: -0.0325320, 12.2178364, 0.0931957, 12.2260962, -10.5326309, 10.4106178
28: 0.5747828, 12.6529541, 0.7342110, 12.6587601, -11.6722565, 11.5000458
29: -0.5571842, 9.0026817, -0.4694778, 9.0076103, -6.8119698, 6.7152519
30: -4.1582403, 9.9701719, -4.0171251, 9.9802475, -12.8128967, 12.6685410
31: -3.2015047, 11.8161869, -3.0728586, 11.8195419, -11.5408554, 11.3907280
32: -19.0427380, -5.8068895, -19.0465889, -5.8397884, -9.3932762, 9.4570656
33: -38.5039444, -16.8375969, -38.4258881, -16.8382072, -16.0160980, 15.8983459
34: -37.8569183, -23.4162254, -37.8608894, -23.4223137, -10.3286819, 10.3435135
35: -29.0738144, -14.0553350, -29.0138397, -14.0562868, -11.8621826, 11.7792702
36: -22.0968552, -9.1901569, -22.0307541, -9.1974268, -9.2752609, 9.2209702
37: -39.7686348, -18.9878845, -39.6342392, -18.9834328, -15.8435974, 15.6972656
38: -36.0741425, -19.3111839, -36.0244675, -19.3379898, -14.5492325, 14.4787369
39: -38.4411888, -16.9060574, -38.3835068, -16.9159260, -14.8543854, 14.8141022
40: -34.3669739, -20.4205208, -34.3806534, -20.4564648, -8.6623573, 8.7300415
41: -21.2283554, -5.2836289, -21.2096100, -5.2944241, -12.3892136, 12.3988991
42: -23.4801559, -11.3927059, -23.4700127, -11.3945065, -9.8903732, 9.8540039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=20, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A2_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3441629, upper bound: 6.2987540
time: 19.89 seconds

## Relational analysis of IS_A2_A2_A2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A2_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3345369
time: 38.29 seconds

## BFS IS instance: IS_A2_A2_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -26.9762344, -9.3821278, -26.9890594, -9.4901371, -12.0023956, 12.0931282
1: -9.7900171, 0.0636883, -9.7964497, -0.0100601, -6.8269482, 6.8904667
2: -4.3732266, 4.9205666, -4.3724885, 4.8844671, -6.8791275, 6.9400597
3: -13.4833488, -0.4982052, -13.4923944, -0.6290805, -9.6021996, 9.7461739
4: -5.4944429, 7.4662247, -5.5016551, 7.3375416, -8.9591599, 9.0810242
5: -8.9363813, 4.3172617, -8.9424191, 4.1934562, -11.4853516, 11.6225510
6: -24.1146336, -8.9038639, -24.1172752, -8.9178600, -9.8180695, 9.8597221
7: -9.6940250, 2.7604594, -9.6959686, 2.6893468, -8.8997650, 8.9717789
8: -12.3395729, 3.1584332, -12.3432198, 3.1004970, -9.4511986, 9.4986954
9: -7.0762920, 8.7950983, -7.0913458, 8.6215801, -10.8816528, 11.0554771
10: -7.0498290, 7.3588099, -7.0651379, 7.1756792, -11.5580978, 11.7593575
11: -4.7854624, 5.0644217, -4.6723757, 5.0643368, -8.3067818, 8.1870384
12: -16.9077301, -0.5602487, -16.8737087, -0.5855522, -11.4421387, 11.4746552
13: -21.4044933, -2.9669342, -21.3839245, -3.0941429, -14.4773178, 14.6251297
14: -22.9464626, -5.0212078, -22.9050961, -5.0386143, -16.9018173, 16.8987045
15: -9.0501709, 3.5935652, -9.0544281, 3.4736812, -9.8440895, 9.9661942
16: -9.6418591, 1.2480116, -9.6545792, 1.1398842, -10.0952682, 10.2282906
17: -20.8899479, -4.1504350, -20.8140717, -4.1603141, -13.7862473, 13.7288437
18: -3.2973919, 11.8141146, -3.2246511, 11.8054276, -11.2030106, 11.1348839
19: 1.7272758, 11.0839195, 1.8475170, 11.0926857, -9.3160286, 9.1890450
20: -0.8938582, 9.8816757, -0.8052025, 9.8887749, -10.7826328, 10.6868782
21: 0.6262352, 13.1124649, 0.7511153, 13.1212225, -12.3943024, 12.2283249
22: 1.9007559, 12.2909746, 1.9846249, 12.2965469, -8.4777412, 8.3840675
23: -0.0008345, 11.0847092, 0.2062470, 11.0994205, -9.7053375, 9.4804459
24: -5.5508208, 9.5101118, -5.3659277, 9.5210066, -12.2295837, 12.0288353
25: -4.5587158, 9.7442236, -4.3893414, 9.7541742, -12.0394974, 11.8604622
26: 2.8355346, 16.2985687, 2.9807062, 16.3061562, -13.4706211, 13.3178625
27: -0.0347598, 12.2159529, 0.0932887, 12.2253904, -10.5351143, 10.4087257
28: 0.5773728, 12.6492739, 0.7342281, 12.6576519, -11.6686630, 11.5136948
29: -0.5594401, 9.0098896, -0.4692420, 9.0091467, -6.8154678, 6.7321453
30: -4.1639223, 9.9736404, -4.0169516, 9.9812584, -12.8211060, 12.6817551
31: -3.2071180, 11.8191109, -3.0726004, 11.8207703, -11.5451012, 11.4062500
32: -19.0450802, -5.8089170, -19.0466576, -5.8399782, -9.3974266, 9.4620934
33: -38.5060196, -16.8385563, -38.4261551, -16.8389645, -16.0148087, 15.8985672
34: -37.8679771, -23.4119377, -37.8604889, -23.4206505, -10.3345528, 10.3238907
35: -29.0718765, -14.0590677, -29.0141640, -14.0578909, -11.8595352, 11.7759132
36: -22.0816574, -9.2032299, -22.0307579, -9.2013302, -9.2823296, 9.2154312
37: -39.7673569, -18.9897652, -39.6345291, -18.9843712, -15.8311768, 15.6939545
38: -36.0676842, -19.3207054, -36.0240707, -19.3411522, -14.5258026, 14.4700851
39: -38.4333878, -16.9162006, -38.3829994, -16.9186668, -14.8139114, 14.8056412
40: -34.3597107, -20.4333706, -34.3782043, -20.4570198, -8.6608086, 8.7155170
41: -21.2316952, -5.2823887, -21.2098236, -5.2943702, -12.3913651, 12.4079552
42: -23.4891834, -11.3865881, -23.4711895, -11.3932457, -9.8809128, 9.8597946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=20, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A2_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3055784
time: 8.95 seconds

## Relational analysis of IS_A2_A2_A2_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3413632
time: 30.82 seconds

## BFS IS instance: IS_A2_A2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -26.9918442, -9.3600988, -26.9937134, -9.4896603, -12.0154228, 12.1196404
1: -9.7960129, 0.0747843, -9.7977858, -0.0096970, -6.8352795, 6.9046936
2: -4.3773928, 4.9265494, -4.3736253, 4.8846526, -6.8837509, 6.9471645
3: -13.4966841, -0.4802299, -13.4962044, -0.6289487, -9.6133614, 9.7685814
4: -5.5120153, 7.4836040, -5.5065804, 7.3376579, -8.9745560, 9.1034126
5: -8.9507589, 4.3327136, -8.9464598, 4.1935520, -11.4975967, 11.6422882
6: -24.1154766, -8.9030876, -24.1171799, -8.9176044, -9.8193893, 9.8631802
7: -9.7027893, 2.7719364, -9.6983356, 2.6894588, -8.9070244, 8.9856911
8: -12.3523006, 3.1697078, -12.3465242, 3.1005583, -9.4634933, 9.5127068
9: -7.0980258, 8.8200359, -7.0975933, 8.6217079, -10.8999252, 11.0866890
10: -7.0783124, 7.3910255, -7.0732994, 7.1759176, -11.5808258, 11.7995567
11: -4.7948370, 5.0741611, -4.6725755, 5.0668182, -8.3197289, 8.1968002
12: -16.9188404, -0.5520569, -16.8738117, -0.5832691, -11.4547691, 11.4826050
13: -21.4143925, -2.9560270, -21.3841476, -3.0913782, -14.4814415, 14.6365013
14: -22.9565506, -5.0086250, -22.9065819, -5.0350485, -16.9113693, 16.9101562
15: -9.0715752, 3.6121535, -9.0599403, 3.4738953, -9.8637466, 9.9904289
16: -9.6569290, 1.2721720, -9.6590099, 1.1402969, -10.1080475, 10.2568855
17: -20.9117813, -4.1295137, -20.8148651, -4.1543665, -13.8141174, 13.7459106
18: -3.2983322, 11.8146610, -3.2249660, 11.8048725, -11.2050629, 11.1364174
19: 1.7159610, 11.0935555, 1.8471434, 11.0954466, -9.3299751, 9.1985817
20: -0.9043694, 9.8878202, -0.8060219, 9.8905840, -10.7949533, 10.6938419
21: 0.6151571, 13.1221542, 0.7503178, 13.1236181, -12.4120789, 12.2396011
22: 1.8884449, 12.2977495, 1.9841418, 12.2984810, -8.4920311, 8.3902493
23: -0.0213928, 11.1038284, 0.2059623, 11.1048965, -9.7312927, 9.4968414
24: -5.5671272, 9.5252600, -5.3661795, 9.5252342, -12.2500076, 12.0421791
25: -4.5787044, 9.7629871, -4.3897867, 9.7594681, -12.0647659, 11.8770790
26: 2.8179722, 16.3076515, 2.9797349, 16.3085384, -13.4905663, 13.3279171
27: -0.0472305, 12.2268696, 0.0929432, 12.2285280, -10.5464325, 10.4159203
28: 0.5594339, 12.6665630, 0.7336943, 12.6623840, -11.6910095, 11.5299263
29: -0.5706520, 9.0172586, -0.4695880, 9.0111694, -6.8289948, 6.7389717
30: -4.1790762, 9.9877234, -4.0175467, 9.9850368, -12.8384094, 12.6945267
31: -3.2175326, 11.8288488, -3.0731111, 11.8230019, -11.5601044, 11.4161644
32: -19.0482521, -5.8056612, -19.0469933, -5.8394346, -9.4008865, 9.4664192
33: -38.5151749, -16.8285809, -38.4264870, -16.8362637, -16.0273743, 15.9084358
34: -37.8715057, -23.4060516, -37.8612442, -23.4195671, -10.3384171, 10.3340073
35: -29.0811272, -14.0465927, -29.0142479, -14.0545292, -11.8709183, 11.7876015
36: -22.0989418, -9.1869011, -22.0308647, -9.1965933, -9.2989502, 9.2270985
37: -39.7842560, -18.9760532, -39.6347008, -18.9805260, -15.8520584, 15.7065201
38: -36.0777321, -19.3107567, -36.0242348, -19.3375397, -14.5384827, 14.4792671
39: -38.4442291, -16.9063187, -38.3832512, -16.9158783, -14.8245392, 14.8131714
40: -34.3708725, -20.4210243, -34.3812103, -20.4567490, -8.6697197, 8.7330112
41: -21.2341671, -5.2779303, -21.2102547, -5.2934055, -12.3954468, 12.4135475
42: -23.4887505, -11.3848162, -23.4704342, -11.3927956, -9.8860779, 9.8611488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=65, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=20, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 824

## Relational analysis of IS_A2_A2_A2_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3083764
time: 26.08 seconds

## Relational analysis of IS_A2_A2_A2_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3441615
time: 21.58 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 49.91 seconds
IS_A1_A2_A1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3412807, upper bound: 6.2747062
IS_A1_A2_A1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3412807, upper bound: 6.3104903
IS_A1_A2_A2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3413645, upper bound: 6.2910660
IS_A1_A2_A2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3268478
IS_A2_A2_A1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3090601
IS_A2_A2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3448448
IS_A2_A2_A1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3090601
IS_A2_A2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3413645, upper bound: 6.3448448
IS_A2_A2_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2692902
IS_A2_A2_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3050736
IS_A2_A2_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2720883
IS_A2_A2_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3078721
IS_A2_A2_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2789102
IS_A2_A2_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3146970
IS_A2_A2_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2817079
IS_A2_A2_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3174948
IS_A2_A2_A2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3441629, upper bound: 6.2959553
IS_A2_A2_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3317381
IS_A2_A2_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3441629, upper bound: 6.2987540
IS_A2_A2_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3345369
IS_A2_A2_A2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3055784
IS_A2_A2_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3413632
IS_A2_A2_A2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3083764
IS_A2_A2_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 49.91
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3441615

## BFS IS instance: IS_A2_A2_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -26.9801750, -9.4068775, -26.9828472, -9.4917936, -12.0024300, 12.0550842
1: -9.7916145, 0.0474508, -9.7970200, -0.0109963, -6.8227253, 6.8665104
2: -4.3740306, 4.9135661, -4.3729558, 4.8840122, -6.8787613, 6.9317207
3: -13.4826183, -0.5193257, -13.4887276, -0.6297214, -9.6001205, 9.7208405
4: -5.4894643, 7.4396625, -5.4907761, 7.3364425, -8.9540138, 9.0460472
5: -8.9342184, 4.2973928, -8.9349136, 4.1930308, -11.4840469, 11.5950775
6: -24.1241150, -8.9030857, -24.1177330, -8.9184628, -9.8219414, 9.8586273
7: -9.6921101, 2.7413068, -9.6942701, 2.6886244, -8.8979111, 8.9497452
8: -12.3393936, 3.1460512, -12.3372879, 3.1002514, -9.4533710, 9.4885216
9: -7.0685973, 8.7582226, -7.0792165, 8.6208897, -10.8734818, 11.0063553
10: -7.0419221, 7.3151908, -7.0439134, 7.1753864, -11.5522766, 11.6950073
11: -4.7751770, 5.0613694, -4.6699080, 5.0606909, -8.2944412, 8.1831169
12: -16.8904495, -0.5622107, -16.8751564, -0.5911257, -11.4191132, 11.4743271
13: -21.3865166, -2.9692059, -21.4022503, -3.1009989, -14.4698753, 14.6306534
14: -22.9410667, -5.0210457, -22.9025440, -5.0494499, -16.8972321, 16.8869324
15: -9.0407295, 3.5688720, -9.0377941, 3.4720464, -9.8321381, 9.9283524
16: -9.6382084, 1.2075815, -9.6424751, 1.1384892, -10.0924911, 10.1744461
17: -20.8591480, -4.1566691, -20.8123913, -4.1768146, -13.7397766, 13.7228470
18: -3.2938902, 11.8065281, -3.2233462, 11.8165150, -11.2128448, 11.1214447
19: 1.7428038, 11.0800381, 1.8481116, 11.0896540, -9.2942810, 9.1845703
20: -0.8865132, 9.8848076, -0.8038621, 9.8892231, -10.7757359, 10.6886692
21: 0.6307285, 13.1189737, 0.7521837, 13.1211929, -12.3686447, 12.2249565
22: 1.9183855, 12.2902441, 1.9856462, 12.2955046, -8.4567757, 8.3811646
23: 0.0285735, 11.0793514, 0.2065375, 11.0895481, -9.6657486, 9.4776459
24: -5.5295982, 9.5060759, -5.3653474, 9.5169868, -12.2056732, 12.0239716
25: -4.5277405, 9.7384701, -4.3888149, 9.7439280, -11.9981766, 11.8555107
26: 2.8523116, 16.3059196, 2.9831457, 16.3131466, -13.4608345, 13.3227739
27: -0.0276334, 12.2169447, 0.0938094, 12.2256479, -10.5361519, 10.4065552
28: 0.5986114, 12.6450062, 0.7348983, 12.6513729, -11.6418915, 11.5093689
29: -0.5438929, 9.0082722, -0.4683975, 9.0090408, -6.7990952, 6.7287102
30: -4.1527443, 9.9735270, -4.0161505, 9.9796495, -12.8184891, 12.6851654
31: -3.1934040, 11.8119965, -3.0719802, 11.8198891, -11.5202255, 11.3985138
32: -19.0490093, -5.8102007, -19.0524940, -5.8411765, -9.3990440, 9.4655418
33: -38.4838867, -16.8476257, -38.4282341, -16.8481445, -15.9903793, 15.8821297
34: -37.8705521, -23.4173164, -37.8603363, -23.4237175, -10.3390007, 10.3079910
35: -29.0508003, -14.0674582, -29.0181160, -14.0692635, -11.8413544, 11.7638550
36: -22.0518112, -9.2101154, -22.0367870, -9.2157116, -9.2500038, 9.2069664
37: -39.7361755, -18.9963417, -39.6321983, -18.9955120, -15.7932281, 15.6861572
38: -36.0455513, -19.3291988, -36.0322456, -19.3524513, -14.4957199, 14.4578171
39: -38.4086876, -16.9238758, -38.3948021, -16.9284534, -14.7835007, 14.7959137
40: -34.3768463, -20.4321556, -34.3727646, -20.4570808, -8.6736526, 8.7047863
41: -21.2328186, -5.2866688, -21.2093906, -5.2975516, -12.3896255, 12.4038582
42: -23.4873314, -11.3894482, -23.4703026, -11.3943920, -9.8739090, 9.8575478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=63, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 657

## Relational analysis of IS_A2_A2_A1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_A1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3350700, upper bound: 6.3448448
time: 22.10 seconds

## Relational analysis of IS_A2_A2_A1_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_A1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3350700, upper bound: 6.3448449
time: 23.70 seconds

## BFS IS instance: IS_A2_A2_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -26.9842968, -9.4063969, -26.9977379, -9.4708786, -12.0265656, 12.0674324
1: -9.7928448, 0.0477905, -9.8021107, -0.0010347, -6.8367729, 6.8747997
2: -4.3749962, 4.9137707, -4.3770108, 4.8895836, -6.8853989, 6.9361687
3: -13.4864140, -0.5192037, -13.5019522, -0.6118159, -9.6224403, 9.7318230
4: -5.4943528, 7.4397802, -5.5082960, 7.3537798, -8.9762955, 9.0613480
5: -8.9382582, 4.2974691, -8.9491558, 4.2084379, -11.5036774, 11.6071434
6: -24.1238823, -8.9028111, -24.1181736, -8.9179592, -9.8252754, 9.8597660
7: -9.6944637, 2.7414370, -9.7029257, 2.6998549, -8.9117470, 8.9565811
8: -12.3423100, 3.1461093, -12.3493614, 3.1104062, -9.4661903, 9.5005703
9: -7.0747671, 8.7583361, -7.1009302, 8.6458340, -10.9046211, 11.0245399
10: -7.0500393, 7.3154287, -7.0722623, 7.2074957, -11.5923500, 11.7176819
11: -4.7754078, 5.0636048, -4.6786766, 5.0700164, -8.3042030, 8.1960068
12: -16.8905506, -0.5601102, -16.8862495, -0.5826726, -11.4268417, 11.4857979
13: -21.3867264, -2.9666576, -21.4118042, -3.0903816, -14.4809990, 14.6316795
14: -22.9425354, -5.0174942, -22.9124031, -5.0368853, -16.9084244, 16.8960266
15: -9.0461731, 3.5690777, -9.0591793, 3.4905753, -9.8562775, 9.9479332
16: -9.6426105, 1.2079720, -9.6574545, 1.1625261, -10.1209373, 10.1870155
17: -20.8599224, -4.1507869, -20.8341370, -4.1559763, -13.7567291, 13.7501755
18: -3.2942212, 11.8059750, -3.2242191, 11.8169584, -11.2142563, 11.1219292
19: 1.7424257, 11.0827560, 1.8369322, 11.0992908, -9.3037758, 9.1984062
20: -0.8873110, 9.8866119, -0.8139353, 9.8953571, -10.7826681, 10.7005472
21: 0.6299388, 13.1212206, 0.7425425, 13.1304893, -12.3798752, 12.2426033
22: 1.9179153, 12.2922096, 1.9734454, 12.3022842, -8.4628983, 8.3953209
23: 0.0283222, 11.0847826, 0.1861024, 11.1085968, -9.6816254, 9.5034180
24: -5.5298142, 9.5102415, -5.3815908, 9.5319872, -12.2188034, 12.0442848
25: -4.5281596, 9.7437153, -4.4087324, 9.7626476, -12.0147018, 11.8806839
26: 2.8513160, 16.3081856, 2.9666505, 16.3214359, -13.4701195, 13.3415356
27: -0.0280044, 12.2199068, 0.0818565, 12.2362413, -10.5435829, 10.4149170
28: 0.5980926, 12.6497173, 0.7170894, 12.6686230, -11.6580048, 11.5315170
29: -0.5442383, 9.0102787, -0.4795185, 9.0163479, -6.8058701, 6.7421436
30: -4.1533065, 9.9773273, -4.0312357, 9.9936895, -12.8311996, 12.7015915
31: -3.1938663, 11.8142014, -3.0819733, 11.8295841, -11.5300751, 11.4132195
32: -19.0491257, -5.8096581, -19.0554333, -5.8379564, -9.4032898, 9.4689369
33: -38.4841995, -16.8449593, -38.4371262, -16.8382854, -16.0001221, 15.8945160
34: -37.8712845, -23.4162483, -37.8638268, -23.4179306, -10.3489037, 10.3118172
35: -29.0509186, -14.0640202, -29.0269508, -14.0570040, -11.8529282, 11.7744789
36: -22.0519238, -9.2054033, -22.0539780, -9.1994171, -9.2616882, 9.2198143
37: -39.7363586, -18.9924984, -39.6490059, -18.9819107, -15.8056107, 15.7069397
38: -36.0457268, -19.3266468, -36.0429344, -19.3396358, -14.5054092, 14.4688759
39: -38.4089470, -16.9211235, -38.4054947, -16.9186592, -14.7915268, 14.8053894
40: -34.3797913, -20.4319019, -34.3838272, -20.4449692, -8.6900482, 8.7133274
41: -21.2331772, -5.2857294, -21.2117577, -5.2931805, -12.3949432, 12.4080582
42: -23.4865398, -11.3889847, -23.4696541, -11.3927088, -9.8751030, 9.8626823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=60, inp2_unstable=63, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=164, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=19, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 580

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 657

## Relational analysis of IS_A2_A2_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_A1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3350700, upper bound: 6.3448448
time: 12.03 seconds

## Relational analysis of IS_A2_A2_A1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_A1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 26, lower bound: -6.3364766, upper bound: 6.3448449
time: 12.26 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -26.9548225, -9.4226494, -26.9852581, -9.4950838, -11.9287186, 12.0450096
1: -9.7736034, 0.0329146, -9.7934084, -0.0145521, -6.7726784, 6.8557301
2: -4.3612642, 4.9032068, -4.3700862, 4.8813081, -6.8569984, 6.9111061
3: -13.4574614, -0.5520713, -13.4881725, -0.6353889, -9.5701675, 9.6879768
4: -5.4603682, 7.4183502, -5.4961939, 7.3337703, -8.9071045, 9.0275917
5: -8.9091969, 4.2674956, -8.9382954, 4.1888256, -11.4535980, 11.5684853
6: -24.1017532, -8.9123268, -24.1170959, -8.9194946, -9.8103027, 9.8443985
7: -9.6739845, 2.7348332, -9.6920090, 2.6845551, -8.8696594, 8.9420547
8: -12.3171797, 3.1342421, -12.3384314, 3.0974374, -9.4202003, 9.4691753
9: -7.0416222, 8.7317781, -7.0850916, 8.6156731, -10.8235893, 10.9867020
10: -7.0211973, 7.3057957, -7.0612459, 7.1720247, -11.5277176, 11.7112465
11: -4.7281294, 5.0343161, -4.6702108, 5.0605483, -8.2491493, 8.1542053
12: -16.8842640, -0.6011994, -16.8732758, -0.5907828, -11.4249878, 11.4324760
13: -21.3600025, -3.0710468, -21.3787918, -3.1101627, -14.4102364, 14.5171890
14: -22.9140778, -5.0503082, -22.9019814, -5.0419931, -16.8482285, 16.8654938
15: -9.0242100, 3.5473404, -9.0516577, 3.4726553, -9.8156319, 9.9200401
16: -9.6229477, 1.2284088, -9.6504889, 1.1387641, -10.0636292, 10.1914825
17: -20.8568649, -4.1896286, -20.8107491, -4.1646504, -13.7543106, 13.6918030
18: -3.2429211, 11.7941017, -3.2141056, 11.8026724, -11.1454468, 11.1101112
19: 1.7781351, 11.0617571, 1.8527129, 11.0900402, -9.2623558, 9.1476288
20: -0.8478463, 9.8670826, -0.7999752, 9.8873596, -10.7352057, 10.6670580
21: 0.6909931, 13.0887890, 0.7568617, 13.1191578, -12.3199692, 12.1944885
22: 1.9335165, 12.2749453, 1.9898796, 12.2935524, -8.4362946, 8.3507462
23: 0.0805829, 11.0463696, 0.2126557, 11.0944099, -9.6192551, 9.4281349
24: -5.4669924, 9.4694138, -5.3584495, 9.5149345, -12.1399002, 11.9810677
25: -4.4999409, 9.7096310, -4.3831692, 9.7487869, -11.9753304, 11.8140450
26: 2.9096766, 16.2705536, 2.9922256, 16.3025665, -13.3928900, 13.2783279
27: 0.0340564, 12.1894646, 0.1008146, 12.2222586, -10.4663849, 10.3784828
28: 0.6437707, 12.6158323, 0.7417028, 12.6530800, -11.5981598, 11.4570160
29: -0.5205840, 8.9860916, -0.4638900, 9.0050812, -6.7725220, 6.6935749
30: -4.0847979, 9.9345188, -4.0088978, 9.9755478, -12.7369003, 12.6259689
31: -3.1433241, 11.7923994, -3.0655921, 11.8169117, -11.4781189, 11.3600273
32: -19.0305901, -5.8337164, -19.0456238, -5.8473139, -9.3745728, 9.4272423
33: -38.4725418, -16.8581848, -38.4250526, -16.8432999, -15.9668350, 15.8774071
34: -37.8522797, -23.4281464, -37.8608360, -23.4263382, -10.3137932, 10.3288879
35: -29.0539684, -14.0746479, -29.0135612, -14.0636139, -11.8212280, 11.7632866
36: -22.0728207, -9.2206345, -22.0318031, -9.2084866, -9.2461319, 9.2061501
37: -39.7145271, -19.0171852, -39.6328049, -18.9881172, -15.7826309, 15.6682663
38: -36.0548019, -19.3470078, -36.0262184, -19.3499317, -14.5094376, 14.4741287
39: -38.4157639, -16.9306278, -38.3821564, -16.9288712, -14.8241425, 14.7941742
40: -34.3524590, -20.4351597, -34.3768311, -20.4565887, -8.6503983, 8.7100983
41: -21.2105217, -5.2990999, -21.2091751, -5.2979364, -12.3714447, 12.3809471
42: -23.4723568, -11.4055758, -23.4707870, -11.3963404, -9.8756027, 9.8415108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=64, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=165, inp2_unstable=164, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=18, inp2_unstable=17, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 671

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 657

## Relational analysis of IS_A2_A2_A2_A1_A1_A1_B1_B1

### Relational analysis result of IS_A2_A2_A2_A1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3363928, upper bound: 6.2692901
time: 27.10 seconds

## Relational analysis of IS_A2_A2_A2_A1_A1_A1_B1_B2

### Relational analysis result of IS_A2_A2_A2_A1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 26, lower bound: -6.3363928, upper bound: 6.2692902
time: 23.11 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 52.48 seconds
IS_A2_A2_A1_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 52.48
Output dim: 26, lower bound: -6.3350700, upper bound: 6.3448448
IS_A2_A2_A1_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 52.48
Output dim: 26, lower bound: -6.3350700, upper bound: 6.3448449
IS_A2_A2_A1_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 52.48
Output dim: 26, lower bound: -6.3350700, upper bound: 6.3448448
IS_A2_A2_A1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 52.48
Output dim: 26, lower bound: -6.3364766, upper bound: 6.3448449
IS_A2_A2_A2_A1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 52.48
Output dim: 26, lower bound: -6.3363928, upper bound: 6.2692901
IS_A2_A2_A2_A1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 52.48
Output dim: 26, lower bound: -6.3363928, upper bound: 6.2692902
IS_A2_A2_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3050736
IS_A2_A2_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2720883
IS_A2_A2_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3078721
IS_A2_A2_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2789102
IS_A2_A2_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3146970
IS_A2_A2_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3440790, upper bound: 6.2817079
IS_A2_A2_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3440790, upper bound: 6.3174948
IS_A2_A2_A2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3441629, upper bound: 6.2959553
IS_A2_A2_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3317381
IS_A2_A2_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3441629, upper bound: 6.2987540
IS_A2_A2_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3345369
IS_A2_A2_A2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3055784
IS_A2_A2_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3413632
IS_A2_A2_A2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3083764
IS_A2_A2_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 52.48
Output dim: 26, lower bound: -6.3441629, upper bound: 6.3441615

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 23.46 + 1781.43 = 1804.89 seconds
